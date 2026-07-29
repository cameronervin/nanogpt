# Evaluation

Evaluation answers two different questions: whether the implementation is
correct, and whether one training stage improved the intended behavior. Passing
a software test is not evidence of model quality; a benchmark score is not
evidence that the implementation is causally correct or reproducible.

The scaffold includes evaluation primitives and software tests. The calibrated
three-stage evaluation CLI and release-gate report described below are planned
for the cloud implementation phase and are not yet exposed by `nanogpt`.

## Immutable evaluation inputs

Every report records the checkpoint, tokenizer, dataset/generator, split, and
resolved-evaluation-config hashes. Evaluation examples are ordered by stable ID
and capped after ordering, never by whichever records arrive first from a
stream. Fixed seeds control sampling. FineWeb validation, GSM8K test, and the
reserved synthetic evaluation are never used for tuning or early data mixing.

The report compares all available stages and includes environment, parameter
count, tokens/steps trained, wall time, measured throughput, and estimated cost.
Missing metrics are explicit `not_applicable` values rather than zeros.

## Base model

Required measurements on a fixed held-out FineWeb-Edu token set:

- token-weighted cross-entropy;
- perplexity `exp(loss)` when numerically representable;
- bits per UTF-8 byte, which is more comparable across tokenizers;
- tokens/second, peak memory, and estimated model FLOPS utilization; and
- fixed-seed completion samples from a small, versioned prompt set.

Base release requires finite validation loss, a decrease from the first
evaluation, non-empty generations, and no causal-mask or checkpoint-resume test
failures. Samples are reviewed as diagnostics, not given a subjective pass/fail
quality score.

## SFT model

SFT reports assistant-token loss, response length, rejected/truncated example
counts, and chat-format validity. A completion is format-valid only if it emits
one assistant response terminated by `<|end|>` before EOS/context exhaustion
without leaking raw role markers into content. The fixed format suite must reach
at least 95% valid termination.

Software verification additionally proves that system, user, and padding
targets are `-100`, assistant content and its terminator are selected, and
prompt tokens contribute zero gradient to the loss. Exact match on reserved
reasoning tasks is reported as the baseline for GRPO.

## GRPO model

Reasoning evaluation uses deterministic greedy pass@1 as the primary score.
Sampled pass@k is secondary and records temperature, top-k, seeds, and `k`.
Exact numeric normalization is identical to training reward parsing, while
examples, templates, and numeric ranges are held out.

Report:

- exact-match pass@1/pass@k;
- mean reward and malformed-answer rate;
- mixed- and zero-variance group rates from training;
- reference KL, clip fraction, entropy, and completion length; and
- SFT-to-GRPO change on the fixed synthetic evaluation.

The v1 release gate is an improvement of at least 10 percentage points over
SFT on reserved synthetic reasoning. GSM8K test pass@1 is always reported but
is not a release gate for a 50–100M parameter model.

## Contamination and interpretation

FineWeb can contain benchmark text or near-duplicates. A benchmark result must
not be presented as uncontaminated without a separate contamination study.
GSM8K test remains excluded from all training, but related solutions may occur
in web pretraining. Synthetic generators reduce dependence on a public
benchmark, yet template leakage is still possible; this is why evaluation uses
disjoint template versions and numeric ranges.

Expected limitations include factual unreliability, weak long-context behavior,
poor instruction generalization, fragile arithmetic, and unsafe generations.
The final report should describe these plainly rather than compare the model to
production assistants.

## Machine-readable report

The planned evaluation command will write JSON plus a human-readable Markdown
summary. The JSON will have one entry per stage with checkpoint/config/data
hashes and metric name, value, unit, example count, and uncertainty where
applicable. A `gates` object will contain every release criterion and an
evidence pointer. The command will exit nonzero when a required gate is false
or lacks evidence.
