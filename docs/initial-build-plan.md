# Nano-GPT: Base, SFT, and GRPO Build Plan

Status: Initial scaffold implemented; production cloud training phases remain planned  
Prepared: 2026-07-29  
Owner decisions: cloud spend below $100, FineWeb-Edu pretraining, GRPO with verifiable reasoning rewards

## 1. Summary

Build an educational, from-scratch decoder-only Transformer in PyTorch and train three reproducible checkpoints:

1. **Base model** — next-token pretraining on a deterministic 2B-token subset of FineWeb-Edu.
2. **SFT model** — assistant-style supervised fine-tuning with assistant-only loss.
3. **RL model** — GRPO post-training on verifiable arithmetic and grade-school math.

The implementation should stay close to Karpathy's readable style while adopting stable modern components: pre-normalized RMSNorm, RoPE, grouped-query causal attention, QK normalization, SwiGLU, fused scaled-dot-product attention, BF16, gradient accumulation, compilation, resumable checkpoints, and deterministic data manifests. The code will implement the model and training loops directly rather than hiding them behind a pretrained model framework.

The primary profile is approximately 101M parameters and 2B pretraining tokens. A throughput probe must select the approximately 51M/1B-token fallback if the projected full pipeline would exceed the hard cost guard.

### Goals and success criteria

- A fresh cloud machine can reproduce tokenizer → base → SFT → GRPO from versioned configs and a single runbook.
- Every stage consumes the previous stage's checkpoint and writes immutable weights plus enough trainer state to resume safely.
- The full cloud bill, including data preparation, evaluation, retry reserve, and short-lived storage, remains below $100.
- The base run has finite, decreasing held-out loss and produces non-empty coherent completions.
- SFT reaches at least 95% valid chat-template termination on a fixed format test set and never trains on system/user/padding tokens.
- GRPO improves exact-match accuracy by at least 10 percentage points over SFT on the reserved synthetic reasoning evaluation. GSM8K test accuracy is reported, but is not a release gate for a model this small.
- Unit and integration tests cover causality, tensor shapes, cache equivalence, loss masking, deterministic sharding, exact checkpoint resume, reward parsing, and the GRPO objective.
- Public modules have docstrings and type hints. Inline comments explain tensor shapes, numerical-stability choices, masking, distributed behavior, and RL math—not obvious syntax.

### Non-goals

- Matching a production assistant or a frontier model.
- Multilingual, multimodal, tool-use, retrieval, serving infrastructure, or safety certification.
- PPO with a learned reward/value model, preference-data collection, LoRA, model parallelism, or multi-node training in v1.
- Reimplementing experimental nanochat optimizations such as Muon, FP8, value embeddings, learned residual scalars, or sliding-window attention before the stable baseline is measured.
- Claiming that open-web data is free from copyrighted, toxic, private, or biased text.

## 2. Requirements and stage contracts

### User stories

- **US-01 Base:** As an individual learner, I can prepare licensed, revision-pinned FineWeb-Edu shards and train a base language model from random initialization.
- **US-02 SFT:** I can initialize from the base checkpoint and teach the same model to follow a documented chat format.
- **US-03 RL:** I can initialize from SFT and run critic-free group-relative policy optimization using deterministic, automatically checkable rewards.
- **US-04 Reproduction:** I can smoke-test on an M1 Mac, resume interrupted cloud runs, and see estimated spend before and during every paid stage.
- **US-05 Understanding:** I can read the implementation and companion docs to understand architecture, tensor shapes, losses, data lineage, checkpoints, and tradeoffs.

### Immutable stage handoff

```text
FineWeb-Edu revision + tokenizer manifest
                    │
                    ▼
          base checkpoint + metrics
                    │
                    ▼
           SFT checkpoint + metrics
                    │
                    ▼
          GRPO checkpoint + metrics
```

Each checkpoint directory must contain:

- `model.safetensors`: model parameters only.
- `model_config.json`: the exact architecture.
- `trainer_state.pt`: trusted-local optimizer, scheduler, scaler, RNG, and sampler state required for resume.
- `run_manifest.json`: stage, parent checkpoint hash, tokenizer hash, dataset revision and split hashes, package-lock hash, Git commit, command, token/step counts, device metadata, wall time, and estimated cost.
- `metrics.jsonl`: append-only structured metrics.

Writes must use a temporary directory and atomic rename. A loader must reject a checkpoint when the config, tokenizer, parent checkpoint, or dataset fingerprint is incompatible.

### Core Python interfaces

```python
@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    max_seq_len: int
    n_layers: int
    d_model: int
    n_query_heads: int
    n_kv_heads: int
    ffn_hidden_size: int
    rope_theta: float
    rms_norm_eps: float
    tie_embeddings: bool
    qk_norm: bool


@dataclass
class ModelOutput:
    logits: Tensor | None
    loss: Tensor | None
    kv_cache: "KVCache | None"


class GPT(nn.Module):
    def forward(
        self,
        input_ids: Tensor,
        targets: Tensor | None = None,
        loss_mask: Tensor | None = None,
        kv_cache: "KVCache | None" = None,
    ) -> ModelOutput: ...


RewardFn = Callable[[str, object], float]
```

Unknown configuration fields must fail validation. All run configs resolve to a canonical JSON representation that is hashed into the run manifest.

### Text and conversation schemas

- FineWeb record: `{"id": str, "text": str, "source_revision": str}`.
- Conversation record: `{"id": str, "messages": [{"role": "system|user|assistant", "content": str}], "source": str, "license": str}`.
- GRPO prompt: `{"id": str, "messages": [...], "answer": str, "reward_type": "numeric_exact"}`.
- No Python `eval`, shell execution, or model-graded reward is permitted.

Chat serialization is fixed before SFT:

```text
<|bos|><|system|>...<|end|><|user|>...<|end|><|assistant|>...<|end|>
```

The 32,768-token vocabulary includes the seven reserved tokens `<|bos|>`, `<|eos|>`, `<|pad|>`, `<|system|>`, `<|user|>`, `<|assistant|>`, and `<|end|>`. IDs are recorded in the tokenizer manifest and cannot change after base training.

## 3. Proposed project scaffolding

The implemented initial scaffold uses this project layout:

```text
nano-gpt/
├── README.md
├── LICENSE
├── pyproject.toml
├── uv.lock
├── .env.example
├── .gitignore
├── configs/
│   ├── model/{tiny,gpt_50m,gpt_100m}.yaml
│   ├── data/{fineweb_edu,sft,grpo}.yaml
│   └── run/{local_smoke,cloud_probe,cloud_100}.yaml
├── docs/
│   ├── initial-build-plan.md
│   ├── architecture.md
│   ├── data-and-licenses.md
│   ├── training-stages.md
│   ├── evaluation.md
│   ├── cloud-budget.md
│   └── troubleshooting.md
├── src/nanogpt/
│   ├── config.py
│   ├── model/          # RMSNorm, RoPE, attention, MLP, blocks, GPT
│   ├── data/           # source adapters, tokenizer, shards, chat packing
│   ├── training/       # shared loop, base, SFT, GRPO, distributed helpers
│   ├── inference/      # sampling and KV cache
│   ├── evaluation/     # loss/BPB, chat format, reasoning exact match
│   └── commands/       # prepare, train, eval, chat entry points
├── scripts/
│   ├── run_local_smoke.sh
│   └── run_cloud_100.sh
├── tests/
│   ├── unit/
│   └── integration/
└── artifacts/          # gitignored runtime data; layout shown below
```

Runtime data is excluded from Git and rooted at `$NANOGPT_HOME` (default `./artifacts`):

```text
artifacts/
├── tokenizer/
├── datasets/{base,sft,grpo}/
├── checkpoints/{base,sft,grpo}/
├── evaluations/
└── logs/
```

## 4. Architecture specification

### Default profiles

| Profile | Layers | Width | Query/KV heads | Head dim | SwiGLU hidden | Context | Vocab | Approx. parameters | Token target |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `tiny` | 4 | 256 | 4 / 2 | 64 | 704 | 256 | 32,768 | 11M | smoke only |
| `gpt_50m` | 12 | 512 | 8 / 2 | 64 | 1,408 | 1,024 | 32,768 | 51M | 1B |
| `gpt_100m` | 12 | 768 | 12 / 4 | 64 | 2,048 | 1,024 | 32,768 | 101M | 2B |

The default `gpt_100m` intentionally resembles GPT-2 small in width and depth while using a smaller custom vocabulary, tied embeddings, and modern blocks. It is not checkpoint-compatible with GPT-2.

### Decoder block

For hidden state `x`, each block is:

```text
x = x + GQA(QKNorm(RMSNorm(x)), RoPE, causal=True)
x = x + SwiGLU(RMSNorm(x))
```

Implementation decisions:

- Decoder-only, causal next-token prediction.
- Pre-normalization with learnable RMSNorm scale and `eps=1e-6`.
- RoPE on all query/key head dimensions, `theta=10,000`, with a registered non-persistent cache sized to the configured context.
- Grouped-query attention with 12 query/4 KV heads in the primary profile; PyTorch SDPA is the default kernel, with a tested MPS/math fallback.
- Parameter-free per-head RMS QK normalization before SDPA to reduce attention-logit instability.
- SwiGLU feed-forward layer, with hidden sizes rounded for efficient matrix multiplication.
- Bias-free linear layers, dropout `0.0`, tied token embedding/language-head weights.
- FP32 master weights and optimizer state; BF16 forward/backward on supported CUDA. MPS smoke tests default to FP32.
- Normal initialization with residual output projections scaled by `1/sqrt(2 * n_layers)`.
- Cross-entropy computed in FP32. Targets use `-100` only for ignored tokens.

Use `torch.nn.functional.scaled_dot_product_attention` rather than materializing an attention matrix. Enable `torch.compile` only after eager-mode correctness tests pass, and compile the model before wrapping it in DDP. Activation checkpointing is configurable but off for the 100M H100 profile unless the throughput probe demonstrates a benefit.

### Inference

- Autoregressive generation supports greedy, temperature, top-k, EOS termination, deterministic seeds, and batched prompts.
- A preallocated per-layer KV cache stores only KV heads, not repeated query-head copies.
- Cache and uncached logits must agree within dtype-appropriate tolerance.
- Generation refuses prompts longer than `max_seq_len` and stops before the cache exceeds context.

## 5. Data and training design

### Tokenizer and FineWeb-Edu

- Pin the `HuggingFaceFW/fineweb-edu` dataset revision and use its `sample-10BT` configuration as the source pool.
- Train a byte-level BPE tokenizer with a 32,768-token vocabulary on a deterministic sample of at least 1B characters. Preserve arbitrary UTF-8 through byte fallback.
- Scaffold boundary: the checked-in pure-Python reference BPE trainer is guarded at 1 MB and is not viable for this production sample. An accelerated deterministic backend plus artifact-compatibility and throughput validation remains required before cloud preparation.
- Assign documents to training or validation by a stable hash of source ID; reserve 0.5% for validation before token packing.
- Insert BOS at each document boundary and EOS after each document. Never split one document across train/validation.
- Pretokenize to checksum-addressed shards of unsigned 16-bit token IDs. Store shard count, token count, min/max ID, source revision, tokenizer hash, and SHA-256 in `manifest.json`.
- Training reads local shards with deterministic rank/worker partitioning and resumes at the exact shard, offset, epoch, and RNG state. Remote streaming is a preparation option, not a dependency of the paid training loop.
- Record ODC-By attribution and Common Crawl terms. Document that FineWeb-Edu can still contain harmful, biased, private, or copyrighted material.

### Base training

- Objective: causal next-token cross-entropy over packed 1,024-token rows.
- Optimizer: fused AdamW when available; `betas=(0.9, 0.95)`, `eps=1e-8`, weight decay `0.1`, no decay on norms.
- Primary maximum learning rate: `6e-4`; 2% linear warmup; cosine decay to 10% of maximum; global gradient norm clip `1.0`.
- Global batch: 262,144 tokens. The single-H100 default is micro-batch 16 × sequence 1,024 × accumulation 16.
- Evaluate fixed held-out tokens, save samples, and checkpoint on both token-count and wall-clock intervals.
- Log loss, bits per byte, learning rate, gradient norm, tokens/sec, model FLOPS utilization estimate, peak VRAM, cumulative tokens, wall time, and estimated dollars.

### SFT

- Initialize from the selected base checkpoint; reset optimizer and schedule.
- Data mixture by sampled tokens: 80% allowlisted SmolTalk-generated conversations, 10% reserved GSM8K SFT partition, 10% deterministic synthetic arithmetic/short word problems.
- For SmolTalk, include only records whose source and transitive license are explicitly allowlisted. The generated Smol-Magpie-Ultra, Smol-constraints, Smol-rewrite, and Smol-summarize components are the initial candidates; do not silently ingest mixed-license components.
- Partition GSM8K train by stable ID hash: buckets 0–5 for SFT, 6–9 for GRPO. GSM8K test remains evaluation-only.
- Compute loss only over assistant content and `<|end|>`; system, user, and padding tokens receive `-100` targets.
- Use boundary-aware packing without truncating away the assistant terminator. Reject or separately bucket overlength examples.
- Initial recipe: at most 50M tokens, global batch 65,536 tokens, learning rate `2e-5`, 3% warmup, cosine decay, weight decay `0.01`, gradient clip `1.0`.

### GRPO with verifiable rewards

- Initialize policy from SFT and keep a frozen SFT reference model for KL measurement.
- Training mixture: 50% reserved GSM8K-train prompts and 50% generated arithmetic/short word problems. Synthetic train/eval generators use disjoint seeds, templates, and numeric ranges.
- Generate 8 completions per prompt with temperature `0.8`, top-k `50`, and at most 256 new tokens.
- Reward is `1.0` only when the normalized final numeric answer exactly matches ground truth; otherwise `0.0`. Parsing accepts a documented final-answer marker, signed integers, reduced fractions, and finite decimals. Malformed, missing, non-finite, or ambiguous answers score zero.
- Normalize advantages within each prompt group. Skip zero-variance groups instead of dividing by a near-zero standard deviation.
- Use a clipped policy-ratio surrogate (`epsilon=0.2`) plus reference KL (`beta=0.02`), completion-token masking, one optimizer epoch per fresh rollout batch, AdamW learning rate `5e-6`, and gradient clip `1.0`.
- Initial horizon: at most 500 optimizer steps with 16 prompts per global step. Stop early for held-out reward regression, KL divergence above the configured ceiling, non-finite values, or budget exhaustion.
- Before paid GRPO, run 128 pilot prompts. Proceed only if at least 5% of completions are correct and at least 20% of groups have mixed rewards. If the gate fails, run one bounded synthetic-reasoning SFT warmup and repeat the gate; if it fails again, stop and report sparse-reward blockage rather than spend the RL allocation.

## 6. Cloud cost guard

Reference hardware is one 80GB H100 SXM. Runpod listed it at **$2.99/hour on 2026-07-29**; the run config must take the actual provider rate as input because availability and pricing change.

| Allocation | Maximum |
| --- | ---: |
| Data/tokenizer and cloud probe | $5 |
| Base training | $62 |
| SFT | $8 |
| GRPO | $17 |
| Evaluation/retry reserve | $6 |
| Short-lived storage | $2 |
| **Total** | **$100** |

Budget behavior:

1. Run the entire `tiny` pipeline on the M1 in FP32 with a tiny local fixture.
2. On the target GPU, run a 10M-token compiled throughput and VRAM probe without committing to the full run.
3. Project each stage using measured tokens/sec or rollout/sec plus a 20% safety margin.
4. Select `gpt_100m` only when the projected full pipeline is at most $90 compute. Otherwise select `gpt_50m`; record the automatic decision in the manifest.
5. Every stage receives `hourly_rate_usd` and `stage_budget_usd`. It checkpoints and exits cleanly before its wall-clock allowance. The run script never automatically starts the next paid stage after a failure.
6. Spot/preemptible capacity is allowed only after exact resume is demonstrated. Keep checkpoints on persistent storage and destroy idle GPU instances promptly.

The dollar limit is a hard constraint; the 2B token target is not. A partial, valid checkpoint is preferable to exceeding budget.

## 7. Implementation phases

Each row is one coding-agent thread and one pull-request-sized change.

| Status | Agent loop | Goal | Story/spec refs | Acceptance criteria | Likely touch points | Validation | Docs | Dependencies | Issue |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Planned | feature | Bootstrap packaging, strict typed config loading, artifact layout, lint/type/test commands, and deterministic seeding | US-04, US-05; Core interfaces | `uv sync` succeeds; unknown config fields fail; canonical config hashes are stable; CPU seed test repeats exactly | project metadata, config, test harness | `uv run ruff check .`; `uv run mypy src`; `uv run pytest tests/unit/test_config.py` | README skeleton, contributing commands | None | `feat: bootstrap nano-gpt project` |
| Planned | feature | Train/load the byte-level BPE tokenizer and render the fixed chat template | US-01, US-02; Text schema | UTF-8/byte round trips pass; reserved IDs are stable; save/load is identical; assistant spans are located correctly | tokenizer and chat rendering | tokenizer unit tests plus fixture snapshot | `data-and-licenses.md` tokenizer section | Bootstrap | `feat: add tokenizer and chat format` |
| Planned | feature | Prepare deterministic FineWeb-Edu train/validation shards with manifests and exact resume offsets | US-01, US-04; FineWeb-Edu | no document crosses splits; checksums reproduce; IDs fit `uint16`; two ranks cover data without overlap; interrupted iteration resumes exactly | data adapters and shard loader | fixture integration test; two-process loader test | data lineage and attribution | Tokenizer | `feat: add FineWeb-Edu pipeline` |
| Planned | feature | Implement and test the modern decoder-only GPT architecture | US-01, US-05; Architecture | parameter counts match profiles; future-token mutations cannot affect earlier logits; eager forward/backward is finite; SDPA GQA matches reference attention | model package | model, causality, initialization, and gradient tests | `architecture.md` with equations/shapes | Bootstrap | `feat: implement GPT model` |
| Planned | feature | Add KV-cache generation and atomic, fingerprint-validated checkpointing | US-04; Stage handoff, Inference | cached/uncached logits match; EOS/context stops work; corrupt or mismatched checkpoints fail loudly; resume restores RNG | inference and checkpoint modules | cache equivalence and checkpoint round-trip tests | checkpoint/inference sections | GPT model, Tokenizer | `feat: add inference and checkpoints` |
| Planned | feature | Implement base trainer, optional compile/DDP, metrics, held-out evaluation, and smoke overfit | US-01, US-04; Base training | one batch overfits; loss is finite/decreasing; accumulated update matches large-batch reference; exact resume matches uninterrupted run; 2-process DDP agrees | shared/base training and base evaluation | CPU/MPS smoke; CUDA probe; DDP integration test | `training-stages.md`, `evaluation.md` | Data, GPT, Checkpoints | `feat: train base model` |
| Planned | feature | Build allowlisted conversation/GSM8K/synthetic datasets and assistant-only SFT | US-02; Conversation schema, SFT | source/license fields are required; split hashes are stable; no gradient comes from prompt/pad tokens; tiny SFT learns terminator format | SFT data and trainer | masking, packing, split, and tiny-overfit tests | SFT data/license documentation | Base trainer, Chat format | `feat: add supervised fine-tuning` |
| Planned | feature | Implement numeric rewards, grouped rollout collection, reference KL, and clipped GRPO updates | US-03; GRPO | answer-parser adversarial tests pass; zero-variance groups are skipped; hand-calculated loss matches; frozen reference has no gradients; tiny policy reward improves | GRPO trainer, rewards, reasoning eval | unit math tests; synthetic end-to-end RL test | GRPO derivation and failure modes | SFT, Inference | `feat: add GRPO reasoning stage` |
| Planned | feature | Add cloud throughput probe, profile-selection rule, budget watchdog, persistent resume, and one-command run scripts | US-04; Cloud cost guard | projected profile choice is deterministic; simulated budget expiry checkpoints/exits zero; no next stage starts after failure; manifests show estimated cost | run configs and shell scripts | dry-run and mocked-clock tests; H100 probe | `cloud-budget.md`, troubleshooting | All trainers | `feat: add sub-$100 cloud workflow` |
| Planned | eval | Calibrate fixed evaluations and produce a three-stage comparison report | US-01–US-05; Success criteria | report contains base loss/BPB, SFT format metrics, SFT-vs-RL exact match, GSM8K pass@1, cost, throughput, and config/checkpoint hashes; release gates are machine-checked | evaluation package and report command | full tiny pipeline in CI; cloud artifact audit | README results and `evaluation.md` | All stages | `eval: validate complete training pipeline` |

## 8. Test and evaluation plan

### Unit tests

- Config bounds, divisibility, unknown fields, canonical hashes, and profile parameter counts.
- Tokenizer arbitrary-byte round trips, reserved tokens, chat rendering, and assistant masks.
- RMSNorm/RoPE known values; GQA shape rules; causal leakage; stable initialization; FP32 loss.
- Packed-shard boundaries, deterministic split/shuffle, rank/worker ownership, corrupt checksums, and resume offsets.
- KV-cache equality, deterministic sampling, EOS and context limits.
- SFT prompt/padding masking and overlength behavior.
- Numeric parser cases: commas, signs, fractions, decimals, multiple markers, NaN/Inf, prose injection, and missing answers.
- GRPO advantages, zero variance, clipping, KL, completion mask, frozen reference, and non-finite guard.
- Budget projection, stage deadline, atomic save, and incompatible checkpoint rejection.

### Integration scenarios

1. Overfit a tiny base model on one batch and verify loss falls sharply.
2. Interrupt/resume base training and compare final weights/optimizer/RNG with an uninterrupted run.
3. Train tiny SFT and verify valid assistant termination plus zero prompt-token gradients.
4. Train tiny GRPO on generated addition and verify held-out exact match improves by at least 10 points.
5. Run eager and compiled paths on CUDA; compare loss and gradients within tolerance.
6. Run two-process DDP and verify examples are non-overlapping and reduced metrics match single-process reference.
7. Execute the entire local M1 smoke pipeline without network access, using checked-in tiny fixtures.
8. Simulate preemption and budget expiry at each cloud stage and prove resumable, bounded behavior.

### Stage metrics

| Stage | Required metrics |
| --- | --- |
| Base | train/validation loss, validation BPB/perplexity, tokens/sec, MFU estimate, VRAM, samples, tokens, wall time, estimated dollars |
| SFT | assistant-token loss, format-valid rate, truncation/rejection counts, response length, held-out exact match |
| GRPO | mean reward, pass@1/pass@k, mixed/zero-variance group rate, KL, clip fraction, entropy, completion length, malformed-answer rate |

Use fixed seeds and immutable evaluation splits. Never tune on FineWeb validation, GSM8K test, or the reserved synthetic evaluation.

## 9. Documentation and code-quality contract

- `architecture.md` explains each tensor transformation with symbols and concrete profile shapes, plus why each modern component replaced its GPT-2 equivalent.
- `data-and-licenses.md` records dataset revisions, licenses/terms, filtering, splits, manifests, attribution, and known data risks.
- `training-stages.md` derives causal LM, masked SFT, and GRPO losses and documents parent/child checkpoint flow.
- `evaluation.md` distinguishes training metrics, release gates, benchmarks, contamination risk, and expected limits of a 50–100M model.
- `cloud-budget.md` gives provider-neutral setup, the measured profile-selection formula, storage/resume procedure, and shutdown checklist.
- `troubleshooting.md` covers OOM, NaN, compile fallback, unavailable fused kernels, dataloader stalls, corrupt shards, preemption, sparse rewards, and cost-guard exits.
- Inline comments are mandatory at non-obvious shape changes, causal masks, loss masks, dtype casts, distributed reductions, checkpoint boundaries, and GRPO calculations. Comments must explain **why** and state invariants.
- Every training command supports `--dry-run`, prints the resolved config/parameter count/projected tokens and cost, and requires explicit confirmation or `--yes` before a paid full run.

## 10. Risks and fixed responses

| Risk | Planned response |
| --- | --- |
| 100M model is too weak for GSM8K reward signal | Use synthetic arithmetic in SFT/RL, enforce the pilot reward-density gate, and treat GSM8K as a reported secondary metric. |
| Custom implementation is slower than the budget model | Use the measured H100 probe and deterministic 50M fallback; stop on the stage dollar cap. |
| FineWeb-Edu contains undesirable or legally sensitive text | Preserve source/terms/attribution, document limitations, keep output research-oriented, and avoid claiming the dataset is clean or risk-free. |
| SmolTalk contains mixed upstream licenses | Require per-source license allowlisting and fail closed when provenance/license is absent. |
| GRPO reward hacking | Exact deterministic parsing, malformed-answer zero reward, adversarial parser tests, held-out templates/ranges, and qualitative sample review. |
| Training divergence | BF16 with FP32 loss/master state, QK norm, scaled initialization, gradient clipping, NaN guards, frequent atomic checkpoints, and eager fallback. |
| Cloud interruption or surprise spend | Persistent checkpoints, exact resume test, explicit hourly rate, wall-clock watchdog, per-stage caps, and no automatic retry/next-stage launch. |

## 11. Research basis

Architecture and training choices were checked against current Context7 documentation for PyTorch, Hugging Face Datasets, and TRL, plus these primary sources:

- Karpathy's current [nanochat repository](https://github.com/karpathy/nanochat), [sub-$100 speedrun](https://github.com/karpathy/nanochat/blob/master/runs/speedrun.sh), [GPT implementation](https://github.com/karpathy/nanochat/blob/master/nanochat/gpt.py), and [GSM8K RL implementation](https://github.com/karpathy/nanochat/blob/master/scripts/chat_rl.py).
- [PyTorch scaled dot-product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) and [custom Transformer building blocks](https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html).
- [FineWeb-Edu dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) and [FineWeb paper](https://arxiv.org/abs/2406.17557).
- [RMSNorm](https://arxiv.org/abs/1910.07467), [RoPE](https://arxiv.org/abs/2104.09864), [GQA](https://arxiv.org/abs/2305.13245), [SwiGLU/GLU variants](https://arxiv.org/abs/2002.05202), and [QK normalization](https://arxiv.org/abs/2010.04245).
- [Chinchilla compute-optimal scaling](https://arxiv.org/abs/2203.15556).
- [DeepSeekMath/GRPO](https://arxiv.org/abs/2402.03300) and current [TRL GRPO documentation](https://huggingface.co/docs/trl/grpo_trainer).
- [GSM8K dataset card](https://huggingface.co/datasets/openai/gsm8k) and [SmolTalk dataset card](https://huggingface.co/datasets/HuggingFaceTB/smoltalk).
- [Runpod pricing](https://www.runpod.io/pricing), consulted 2026-07-29 only as a cost reference; actual launch-time pricing is authoritative.

## 12. Assumptions

- The model is English-first and intended for learning/research rather than commercial deployment.
- One rented CUDA GPU is the default; DDP support is implemented and tested but not required for the first full run.
- The M1 MacBook Air is used for editing, tests, and the tiny smoke pipeline, not the full training run.
- The $100 ceiling includes estimated compute and short-lived storage; network egress and taxes must be checked before launch.
- Dataset/model artifacts remain local or in private storage until their documentation and licenses are reviewed.
- Dependency versions are pinned in `uv.lock` when scaffolding is implemented; CI tests the locked versions rather than floating latest packages.
