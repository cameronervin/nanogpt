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

## Next

1. **First real run.** Run `nanogpt probe` on the target GPU, then the
   FineWeb tokenizer and shard preparation. Measure preparation time against
   the $5 data cap.
2. **Multi-GPU (DDP).** `TokenLoader` already partitions by rank. Still
   needed: process-group setup, `DistributedDataParallel` wrapping after
   compile, rank-0-only checkpoints and logs, and reduced metrics.
3. **SFT efficiency.** Length-bucket or pack conversations to cut padding,
   and sample the mixture by assistant tokens rather than by example count.
4. **GRPO throughput.** Batch rollouts across prompts with left padding
   instead of generating one prompt group at a time. Add a token-entropy
   metric.
5. **Pilot fallback.** Add one bounded synthetic-arithmetic SFT warmup when
   the GRPO reward-density gate fails, then re-run the pilot.
6. **Evaluation.** Add GSM8K test pass@1 and sampled pass@k. Add MFU and
   peak-memory logging. Make held-out arithmetic disjoint by construction
   (the training generators currently draw from the same operand range).
7. **Contamination study** before quoting any benchmark number.

## Non-goals (v1)

Matching production assistants. Multilingual, multimodal, tool use or
retrieval. PPO with learned reward or value models. LoRA. Model parallelism.
Experimental optimizers (Muon), FP8 and sliding-window attention before the
stable baseline is measured.
