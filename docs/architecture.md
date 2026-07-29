# Architecture

This project implements a decoder-only Transformer directly in PyTorch. It is
small enough to read end to end, but its default block follows current stable
practice rather than reproducing GPT-2 line for line.

## Profiles and invariants

The versioned profiles live in `configs/model/`. For a batch of token IDs
`input_ids[B, T]`, every profile requires:

- `T <= max_seq_len`;
- `d_model % n_query_heads == 0`;
- `n_query_heads % n_kv_heads == 0`; and
- token IDs in `[0, vocab_size)`.

| Profile | Hidden state | Query heads | KV heads | Head width | Block output |
| --- | --- | ---: | ---: | ---: | --- |
| tiny | `[B,T,256]` | 4 | 2 | 64 | `[B,T,256]` |
| gpt_50m | `[B,T,512]` | 8 | 2 | 64 | `[B,T,512]` |
| gpt_100m | `[B,T,768]` | 12 | 4 | 64 | `[B,T,768]` |

The vocabulary has 32,768 entries and the language-model head shares its
weight with the input embedding. Configuration validation rejects unknown
fields and invalid head ratios before allocating parameters.

## Decoder computation

Tokens are embedded to `x in R^(B x T x d)`. Each block computes

```text
h = RMSNorm(x)
x = x + Attention(QKNorm(h), RoPE, causal=True)
h = RMSNorm(x)
x = x + SwiGLU(h)
```

A final RMSNorm produces hidden states, and multiplication by the transposed
embedding matrix produces logits `[B,T,V]`.

### RMSNorm and pre-normalization

For a hidden vector `x`, RMSNorm is

```text
x_hat = x * rsqrt(mean(x²) + eps)
output = weight * x_hat
```

The reduction and reciprocal square root are evaluated in FP32 when needed for
stability, then cast back. Pre-normalization gives each sublayer normalized
input while leaving a direct residual route for gradients. Unlike GPT-2's
LayerNorm, RMSNorm has no learned bias or mean subtraction.

### Grouped-query causal attention

One bias-free projection produces queries
`[B,Hq,T,Dh]`, keys `[B,Hkv,T,Dh]`, and values `[B,Hkv,T,Dh]`. Each KV head is
shared by `Hq/Hkv` query heads. Sharing reduces KV-cache memory without changing
the query capacity. Q and K receive per-head, parameter-free RMS normalization,
which limits attention-logit growth.

RoPE rotates adjacent pairs of query and key features using position-dependent
angles. Frequencies are derived from `theta=10,000`; the sine/cosine cache is a
non-persistent buffer because it can be reconstructed from configuration. RoPE
replaces GPT-2's learned absolute position embedding and naturally applies to
cached decode positions.

PyTorch scaled-dot-product attention performs

```text
softmax((Q @ K.transpose(-2, -1)) / sqrt(Dh) + causal_mask) @ V
```

without requiring the implementation to materialize a full attention matrix.
The code requests causal behavior from the kernel. A reference math path is
kept for tests and for platforms that cannot use a fused CUDA kernel. A future
token mutation must never change an earlier output logit.

### SwiGLU

The feed-forward network uses two parallel projections:

```text
SwiGLU(x) = down(silu(gate(x)) * value(x))
```

The configured intermediate widths are chosen near the parameter-equivalent
size of a conventional 4x GELU MLP and rounded for matrix-multiplication
efficiency. All projections are bias-free.

Bias-free projections and dropout `0.0` are fixed architecture decisions, not
profile fields. This keeps the strict public `ModelConfig` contract small and
prevents profiles from silently diverging from the tested v1 architecture.

## Initialization and numeric policy

Embeddings and ordinary linear weights use zero-mean normal initialization.
Residual-output projections use the same standard deviation divided by
`sqrt(2 * n_layers)`, limiting residual-stream variance as depth grows.
Dropout is zero in the default small-data-efficient recipes.

The implemented local MPS/CPU smoke uses FP32. The cloud training phase will
use BF16 autocast on supported CUDA while retaining FP32 parameters, optimizer
states, normalization reductions, cross-entropy logits, and reported losses.
The shared update loop already clips gradients to the configured norm and stops
before an optimizer update when loss or gradients are non-finite.

## Loss and masking

Base training predicts the next token: logits at position `t` are compared with
the token at `t+1`. SFT uses the same cross-entropy but replaces targets for
system, user, and padding positions with `-100`; only assistant content and its
terminating `<|end|>` contribute. Loss is the mean over non-ignored target
tokens, not over padded sequence length.

## KV cache and generation

At inference, each layer owns preallocated K and V arrays shaped
`[B,Hkv,max_seq_len,Dh]`. Prefill writes prompt positions once. Every decode
step appends one position and attends over the populated prefix. KV heads remain
grouped in storage rather than being copied to query-head count.

Generation checks the prompt length before prefill and the cache boundary before
every append. Greedy or sampled decoding stops at EOS or the context limit.
Fixed seeds make sampling repeatable. Test coverage compares cached and
uncached logits at every position within dtype-appropriate tolerance.

## Deliberately optional optimizations

The cloud phase may enable `torch.compile` only after an eager correctness
probe; it is not used by the local MPS pipeline. Activation checkpointing and
DDP are also planned rather than implemented in this initial scaffold, and the
budget plan assumes one GPU. FP8, model parallelism, custom kernels,
sliding-window attention, and experimental optimizers are outside v1.
