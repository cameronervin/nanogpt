# Roadmap

## Done

- Modern decoder (RMSNorm, RoPE, GQA, QK-norm, SwiGLU, tied embeddings) with
  a KV cache.
- Strict typed run configs. Every field is consumed, and unknown or mistyped
  keys fail.
- Rust-backed byte-level BPE (`hf`), plus the pure-Python reference with the
  same contract.
- Checksummed `uint16` shards and a memory-mapped, rank-partitioned,
  exactly resumable `TokenLoader`.
- One `Trainer`, providing:
  - BF16 autocast and `torch.compile`;
  - accumulation and the cosine schedule;
  - eval cadence and atomic step checkpoints;
  - bitwise-exact resume (tested on CPU);
  - the wall-clock budget guard.
- Base, SFT (assistant-only loss) and GRPO with real KV-cached rollouts, exact
  numeric rewards, the frozen reference KL and a reward-density pilot gate.
- Throughput probe with cost projection, eval report with release gates, and
  an offline smoke pipeline that exercises all of the above.

- Real-data rehearsal (`configs/runs/rehearsal.yaml`). The FineWeb, GSM8K and
  SmolTalk adapters are verified at their pinned revisions, and preparation
  throughput is measured (see the runbook).
- First cloud run, base through GRPO, for about $26
  ([results](results.md)).
- Carry-format arithmetic answers (one digit per token) and configurable
  operand lengths.
- Sampled pass@k in the GRPO pilot and the eval report, held-out arithmetic
  accuracy during SFT, and `grpo.init_step` to start GRPO from an
  intermediate SFT checkpoint.

## Next

1. **Digit-spaced prompts.** Write operands as `4 7 + 3 8` so the model can
   attempt lengths it was not trained on, and run GRPO on the partly solved
   lengths.
2. **Longer GRPO.** Held-out accuracy and pass@8 were still rising at step
   150 of the first run.
3. **Go/no-go and sweep as commands.** Fold `scripts/experiments/` into the
   CLI: `nanogpt check grpo` for the pass@k gate and a short learning-rate
   sweep.
4. **Faster shard preparation.** Overlap streaming with encoding and encode in
   several processes. Put the data stage under the budget guard like the
   training stages.
5. **Multi-GPU (DDP).** `TokenLoader` already partitions by rank. Still
   needed: process-group setup, `DistributedDataParallel` wrapping after
   compile, rank-0-only checkpoints and logs, and reduced metrics.
6. **SFT efficiency.** Length-bucket or pack conversations to cut padding,
   and sample the mixture by assistant tokens rather than by example count.
7. **Evaluation.** Add GSM8K test pass@1, MFU and peak-memory logging. Make
   held-out arithmetic disjoint from training by construction.
8. **Contamination study** before quoting any benchmark number.

## Non-goals (v1)

Matching production assistants. Multilingual, multimodal, tool use or
retrieval. PPO with learned reward or value models. LoRA. Model parallelism.
Experimental optimizers (Muon), FP8 and sliding-window attention before the
stable baseline is measured.
