# Design

How the model, data, and three training stages fit together, and why each
choice was made. Code-level invariants live in docstrings and tests, not here.

```text
corpus ──► tokenizer ──► base shards ──► base ──► SFT ──► GRPO ──► eval report
                         SFT jsonl ───────────────┘        │
                         GRPO prompts ─────────────────────┘
```

Every arrow is a file under `$NANOGPT_HOME/runs/<run>/`, so each step can be
run, resumed, and inspected on its own (`nanogpt <step> --config ...`).

## Code map

| Path | Responsibility |
| --- | --- |
| `config.py` | `ModelConfig`, typed `RunConfig` + stage configs, strict YAML loader, `RunPaths` |
| `model/` | `layers.py` (RMSNorm, RoPE, GQA, SwiGLU), `gpt.py` (block + GPT), `kv_cache.py` |
| `tokenizer/` | `hf.py` (production, Rust BPE), `reference.py` (readable pure-Python BPE) |
| `data/` | `sources.py` (FineWeb/GSM8K/SmolTalk/synthetic), `shards.py` (writer + `TokenLoader`), `chat.py` (template + SFT collation), `prepare.py` |
| `train/` | `trainer.py` (the one loop), `base.py`, `sft.py`, `grpo.py`, `checkpoint.py`, `budget.py`, `optim.py`, `probe.py` |
| `generate.py`, `rewards.py`, `evaluate.py`, `cli.py` | sampling, exact numeric rewards, metrics + report, commands |

## Model

Decoder-only Transformer, written directly in PyTorch. Profiles live in
`configs/models/` (`nano` for smoke tests, `tiny`, `gpt_50m`, `gpt_100m`).

| Profile | Layers | Width | Q/KV heads | SwiGLU hidden | Context | Params |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `tiny` | 4 | 256 | 4 / 2 | 704 | 256 | 11M |
| `gpt_50m` | 12 | 512 | 8 / 2 | 1,408 | 1,024 | 51M |
| `gpt_100m` | 12 | 768 | 12 / 4 | 2,048 | 1,024 | 101M |

Each block is pre-normalized:

```text
x = x + GQA(QKNorm(RoPE(RMSNorm(x))), causal)
x = x + SwiGLU(RMSNorm(x))
```

- **RMSNorm** instead of LayerNorm: no mean subtraction or bias, and the
  reduction runs in FP32 under mixed precision.
- **RoPE** instead of learned positions: rotates adjacent query/key feature
  pairs by position-dependent angles, and applies directly to cached decode
  positions.
- **Grouped-query attention**: each KV head serves `Hq/Hkv` query heads, which
  shrinks the KV cache. Attention uses `F.scaled_dot_product_attention`, with
  native GQA where available and repeated KV heads on MPS.
- **QK-norm**: parameter-free per-head RMS normalization of Q and K bounds
  attention logits.
- **SwiGLU**: `down(silu(gate(x)) * up(x))`, sized near a 4x GELU MLP's
  parameter count. All projections are bias-free, and dropout is 0.
- **Tied embeddings** and normal(0, 0.02) init. Residual output projections
  are scaled by `1/sqrt(2·n_layers)`.

The loss is cross-entropy in FP32 over targets that are not `-100`, averaged
over supervised tokens. SFT expresses its prompt mask purely through `-100`
targets.

Inference preallocates a per-layer KV cache of shape `[B, Hkv, T, Dh]`.
Cached and uncached logits are tested to match.

## Tokenizer

Byte-level BPE, 32,768 entries for cloud runs. The seven reserved tokens
`<|bos|> <|eos|> <|pad|> <|system|> <|user|> <|assistant|> <|end|>` hold
IDs 0–6, and these IDs never change. Byte fallback round-trips any UTF-8 input.

- `hf` (production) trains and encodes in Rust with GPT-2-style regex
  pre-splitting. The reserved strings are ordinary vocabulary entries rather
  than HF "added tokens", so user text containing `<|assistant|>` cannot
  inject a role marker.
- `reference` shows the BPE algorithm in about 100 lines. It is guarded at
  1 MB of training text because it rewrites the corpus once per merge.

A tokenizer directory holds `tokenizer.json` and `manifest.json` (backend,
sha256, reserved IDs, and for `hf` the library version and training
characters). Loading verifies the checksum.

## Data

**Base:** documents are streamed from the pinned FineWeb-Edu `sample-10BT`
revision, or from an offline fixture. Each document goes to train or
validation by a SHA-256 hash of its ID, before tokenization, so a document
never crosses splits. Documents are wrapped as `<|bos|> text <|eos|>` and
written as little-endian `uint16` shards with per-shard checksums.
`TokenLoader` memory-maps the shards, gives each rank a disjoint subset of
them, packs `[B, T]` batches, wraps into new epochs, and keeps a
`{epoch, shard, offset}` cursor in every checkpoint for exact resume.

**SFT:** the mixture comes from `sft.sources`: allowlisted SmolTalk subsets,
GSM8K train buckets 0–5 and synthetic arithmetic. Each source contributes up
to `examples` conversations. Conversations longer than the context are
rejected, not truncated, and counted in the manifest. The chat template is
fixed:

```text
<|bos|><|system|>…<|end|><|user|>…<|end|><|assistant|>…<|end|>
```

Only assistant content and its `<|end|>` are targets, so the model learns
both the answer and when to stop.

**GRPO:** prompts come only from sources with verifiable numeric answers:
GSM8K train buckets 6–9 and synthetic arithmetic. The GSM8K test split is
evaluation-only.

### Sources and licenses

| Use | Dataset | Revision | Terms |
| --- | --- | --- | --- |
| Base | `HuggingFaceFW/fineweb-edu` `sample-10BT` | `87f09149…` | ODC-By 1.0 + Common Crawl terms |
| SFT/RL | `openai/gsm8k` `main` | `740312ad…` | MIT |
| SFT | `HuggingFaceTB/smoltalk` (4 generated subsets) | `5feaf2fd…` | Verify per-subset license before use |
| SFT/RL/eval | project arithmetic generators | seed in config | Project-generated |

The code license does not cover datasets, weights, or outputs. FineWeb-Edu
is open-web text and may contain copyrighted, harmful, biased, or private
content. Do not redistribute prepared shards.

## Training stages

All three stages share `Trainer`. It handles:

- the warmup + cosine schedule;
- gradient accumulation and clipping;
- BF16 autocast and optional `torch.compile`;
- JSONL metrics, including tokens/s and estimated $;
- periodic evaluation;
- atomic step checkpoints and exact resume;
- the per-stage budget guard.

A stage only supplies a `step_fn(lr)`.

**Base:** next-token loss on packed rows. The cloud recipe is 262,144
tokens per step (16 × 1,024 × 16 accumulation) for 7,630 steps ≈ 2B tokens.
AdamW uses β = (0.9, 0.95), weight decay 0.1 on matrices only, peak LR 6e-4,
2% warmup and cosine decay to 10%.

**SFT:** starts from the finished base checkpoint (its hash is verified and
recorded as the parent) with a fresh optimizer. The loss is
`Σ m_t · CE_t / Σ m_t`, where `m_t` marks assistant targets. LR 2e-5, weight
decay 0.01.

**GRPO:** starts from SFT. A frozen copy of SFT is the KL reference. Each step:

1. For each of `micro_batch_size` prompts, sample `G = 8` completions with
   the KV cache (temperature 0.8, top-k 50). Sampling seeds derive from the
   step, so a resume reproduces the rollouts.
2. Reward each completion 1.0 if its single `<final>…</final>` (or GSM8K
   `####`) answer parses to exactly the ground-truth number, else 0. There is
   no `eval` and no model grading.
3. Advantages: `A_i = (r_i − mean(r)) / std(r)` within the group. Groups with
   zero variance are skipped, never inflated with an epsilon.
4. Loss: `−mean_t min(ρ·A, clip(ρ, 1±ε)·A) + β·KL(π‖π_ref)` over completion
   tokens only, with ε = 0.2 and β = 0.02. There is one optimizer update per
   fresh rollout.

Before the first update, a pilot of `pilot_prompts` prompts must reach at
least 5% correct completions and at least 20% mixed-reward groups. Otherwise
the stage stops instead of spending the RL budget on a zero signal.

## Checkpoints and lineage

```text
checkpoints/<stage>/step_000500/
  model.safetensors   weights only (safe to share after review)
  model_config.json
  trainer_state.pt    optimizer, RNG, step, tokens, data cursor (pickle: trusted-local only)
  run_manifest.json   lineage + resolved config + wall time + estimated cost
checkpoints/<stage>/STATUS.json   {"status": complete|budget_exhausted, "checkpoint": ...}
```

Checkpoints are written to a temporary sibling and renamed atomically, and
never overwritten. On resume, the loader rejects a checkpoint whose model
config, weights checksum, tokenizer hash, dataset hash or parent-model hash
differs.

Other config edits, such as LR or cadence, are allowed on resume and are
recorded in the next manifest.

## Evaluation

`nanogpt eval` writes `eval/report.json`:

- **Base:** held-out loss, perplexity and bits per byte on fixed validation
  batches.
- **SFT and GRPO:** greedy pass@1, format validity and malformed-answer rate
  on held-out arithmetic prompts. The seed is never used for training.

Release gates are:

- finite base loss;
- SFT format validity of at least 95%;
- GRPO exact match at least 10 points above SFT;
- estimated cost within budget.

GSM8K test accuracy is planned as a secondary metric only (see the roadmap).
FineWeb may contain benchmark text, so no score should be presented as
uncontaminated.
