# Troubleshooting

Start from the first error in the log and preserve the resolved config,
manifest, and last valid checkpoint before changing anything. A workaround that
changes model, tokenizer, data ordering, or optimizer semantics is a new run and
must receive a new manifest hash.

## Out of memory

1. Confirm no unrelated process owns GPU memory and the selected profile is the
   one recorded by the probe.
2. Reduce micro-batch size and increase gradient accumulation by the same factor
   to preserve global tokens per update.
3. Retry eager mode if compilation retained excessive buffers.
4. Enable activation checkpointing only after the above; re-run the throughput
   projection because recomputation changes cost.

Do not reduce context length, hidden size, or global batch silently on resume.
Those are configuration changes that require a fresh run or explicit migration.

## NaN or infinite loss/gradients

Stop before the optimizer step. Inspect the first non-finite metric, input token
range, mask denominator, learning rate, gradient norm, and attention logits in
eager FP32. Verify cross-entropy and normalization reductions are FP32, ignored
targets are exactly `-100`, and the SFT mask contains at least one target. Resume
from the last checkpoint only after reproducing the failing batch in a bounded
diagnostic; never keep training past a non-finite update.

## Compile or fused-kernel failure

Run the same batch in eager mode. PyTorch SDPA may select different kernels by
device, dtype, shape, and installation. The math fallback is valid but slower;
record it and repeat the paid cost probe. MPS smoke uses eager FP32 and is not
expected to match H100 throughput or kernel selection. A compiled/eager
correctness test compares loss and gradients before full training.

## Data loader stall or poor throughput

Check that training reads verified local shards rather than a remote stream.
Verify persistent-disk bandwidth, worker count, file-descriptor limit, shard
checksums, and that ranks own disjoint partitions. Excessive tiny shards cause
metadata overhead. When debugging, run one worker and log shard/offset progress;
then restore parallelism and repeat the throughput probe.

## Corrupt shard or manifest mismatch

Do not skip the shard or edit its checksum. Quarantine the entire incomplete
prepared-data directory, reproduce it from the pinned source/tokenizer/config,
and compare manifests. A changed upstream revision, reserved token ID, split
seed, or packing version intentionally invalidates old shards and checkpoints.

## Resume differs from uninterrupted training

Verify optimizer, scheduler, scaler, Python/NumPy/Torch/device RNGs, sampler
epoch, shard index, token offset, accumulation phase, and global step are all
restored before fetching a batch. Compiled and distributed reductions can be
nondeterministic on some devices; prove exact resume on CPU first, then use a
documented tolerance for CUDA. Never infer a cursor from the metric line count.

## Checkpoint rejected

Read the reported fingerprint difference. Architecture, tokenizer, dataset,
resolved run config, and parent checkpoint are safety boundaries. Point to the
correct artifact or begin a new run; do not bypass validation. If an atomic
temporary directory remains after preemption, treat it as incomplete and resume
from the most recent committed sibling.

## GRPO has only zero rewards

Inspect parser failure categories and manually review a small, redacted sample.
Confirm the configured final-answer marker and ground-truth normalization agree.
If the 128-prompt pilot misses the 5% correct or 20% mixed-group gate, run the
one bounded synthetic SFT warmup and repeat. A second failure ends the RL stage;
increasing the budget or inventing epsilon advantages does not fix sparse signal.

## Reward rises but reasoning quality falls

Look for marker spam, multiple answers, malformed fractions, copied prompt
numbers, or template-specific shortcuts. The parser should score ambiguous or
non-finite answers zero. Compare held-out template/range performance, malformed
rate, KL, completion length, and qualitative samples. Stop on held-out reward
regression even if training reward rises.

## Budget guard exit

A clean `budget_exhausted` exit is expected when the stage cap is reached.
Verify the checkpoint and manifest, compare estimated usage with the provider
console, and stop the instance. Do not let a wrapper automatically launch the
next stage. To continue, explicitly supply the same valid resume configuration
and a remaining budget that still fits the overall $100 ceiling.

## M1 local smoke is slow

Confirm the tiny profile, short local fixture, FP32, `num_workers: 0`, and
`compile: false` from `configs/run/local_smoke.yaml`. The smoke test validates
behavior, not performance. It should not download FineWeb-Edu or attempt the
50M/100M profiles. Use CPU if an operation lacks a stable MPS implementation.
