# Data, tokenizer, and licenses

Data preparation is a reproducible build step, not an implicit part of the
paid training loop. Every output shard is tied to a source revision, tokenizer
hash, split algorithm, and checksum in a manifest.

The initial scaffold implements a small deterministic reference byte-BPE
trainer and sharding from local FineWeb-schema JSONL plus offline fixtures.
Direct revision-pinned Hugging Face preparation, an accelerated tokenizer
trainer, and the full billion-character tokenizer sample are the next
cloud-data phase; the sections below define their required contract.

## Source registry

| Use | Dataset | Pinned revision | Terms/license to retain |
| --- | --- | --- | --- |
| Base | `HuggingFaceFW/fineweb-edu`, `sample-10BT` | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` | ODC-By 1.0 attribution plus Common Crawl terms |
| SFT/RL | `openai/gsm8k`, `main` | `740312add88f781978c0658806c59bc2815b9866` | MIT dataset license |
| SFT candidate mixture | `HuggingFaceTB/smoltalk` selected generated configs | `5feaf2fd3ffca7c237fc38d1861bc30365d48ffa` | Verify per-source/transitive license and attribution before inclusion |
| SFT/RL | Project arithmetic generators | generator version and seed in manifest | Project-generated; template provenance must be project-owned |

The revisions above were observed on 2026-07-29 and are encoded in
`configs/data/`. Updating one is a data migration: regenerate manifests and
shards rather than overwriting prior outputs.

The code license does not grant rights to dataset content, trained weights, or
outputs. FineWeb-Edu is derived from open-web crawl data. Educational scoring
and filtering do not guarantee that text is accurate, unbiased, non-toxic,
non-private, or free of copyright restrictions. Do not redistribute prepared
shards. Review the dataset cards and applicable terms before distributing a
checkpoint.

## FineWeb-Edu preparation

1. Stream records from the pinned `sample-10BT` source.
2. Construct a stable identity from the source ID. Missing or duplicate IDs are
   fatal rather than silently replaced by stream order.
3. Hash the ID with the documented seed. Assign 0.5% of whole documents to
   validation before tokenization; a document can never cross splits.
4. Normalize only line endings. Do not apply lossy Unicode cleanup.
5. Add BOS before and EOS after each document, pack tokens into fixed-length
   training rows, and write checksum-addressed `uint16` shards.
6. Verify counts, token range, checksums, and split non-overlap before marking
   the manifest complete.

Remote streaming is permitted during preparation. Training consumes only local
verified shards so a network failure cannot change sample order mid-run.

## Tokenizer contract

The tokenizer is byte-level BPE with vocabulary size 32,768, trained from a
deterministic FineWeb-Edu sample of at least one billion characters. Byte-level
fallback must round-trip arbitrary UTF-8 input without an unknown token.

### Scaffold scalability boundary

`ByteBPETokenizer.train` is intentionally readable pure Python. It recounts
pairs and rewrites the tokenized corpus for every merge, so it is suitable for
unit tests and the offline smoke fixture only. It is not computationally viable
for one billion characters or a full 32,768-entry vocabulary. The reference
trainer and `nanogpt tokenizer` command fail closed above 1,000,000 UTF-8 input
bytes to prevent accidental long-running or paid preparation jobs.

Before cloud data preparation is enabled, add an accelerated, deterministic
byte-level BPE backend and a compatibility test proving that exported token
pieces, merge ranks, reserved IDs, encoding, decoding, and canonical hashes
match this artifact contract. The accelerated run must stream or bound its
sampling memory, record its backend/library version, and demonstrate practical
throughput on the planned billion-character sample. Raising or bypassing the
reference guard is not an acceptable production path.

These tokens are reserved before BPE training and their IDs are immutable:

```text
<|bos|> <|eos|> <|pad|> <|system|> <|user|> <|assistant|> <|end|>
```

`tokenizer.json` is accompanied by `manifest.json` containing the source
revision, sampling algorithm and seed, character count, trainer settings,
reserved-token mapping, library version, file SHA-256, and a canonical
tokenizer hash. A checkpoint loader compares this hash; changing a reserved ID
after base training is incompatible.

The fixed chat format is

```text
<|bos|><|system|>...<|end|><|user|>...<|end|><|assistant|>...<|end|>
```

The renderer returns token IDs and a same-length assistant mask. Only assistant
content and the assistant `<|end|>` are `true`. The role marker is context, not
a target. Packing never truncates away a terminator; an overlength conversation
is rejected or routed to an explicit long-example bucket.

## SFT provenance gate

The initial SmolTalk candidate configs are `smol-magpie-ultra`,
`smol-constraints`, `smol-rewrite`, and `smol-summarize`. A name in the config
is not sufficient evidence of reuse rights. The adapter must fail closed unless
each record has a recognized source and its expected license matches the
allowlist. Save per-source accepted/rejected counts and the exact allowlist in
the prepared-data manifest.

GSM8K training records use a stable ID hash: buckets 0–5 are SFT, buckets 6–9
are GRPO. The published test split is evaluation-only. Synthetic training and
evaluation generators have different versions, seeds, templates, and numeric
ranges to prevent template memorization from becoming the reported result.

## Shard manifest minimum fields

Every prepared dataset directory contains an immutable `manifest.json` with:

- schema and generator versions;
- source names, configurations, revisions, splits, terms, and licenses;
- tokenizer hash and reserved-token mapping;
- split rule/seed and ordered shard list;
- per-shard path, byte size, SHA-256, token/record counts, and min/max token ID;
- total accepted/rejected records and reasons; and
- creation command, package-lock hash, Git commit, and completion flag.

Write to a temporary directory and rename only after verification. Consumers
reject incomplete manifests, changed checksums, token IDs outside the
vocabulary, and a tokenizer mismatch.
