from __future__ import annotations

import json

import pytest
import torch

from nanogpt.data.chat import encode_conversation, render_conversation
from nanogpt.data.fineweb import assign_split
from nanogpt.data.schemas import ConversationRecord, FineWebRecord
from nanogpt.data.sft import collate_sft, require_allowlisted
from nanogpt.data.shards import Cursor, ShardReader, prepare_shards
from nanogpt.data.tokenizer import (
    REFERENCE_TRAINER_MAX_BYTES,
    RESERVED_TOKENS,
    ByteBPETokenizer,
)


def test_byte_bpe_round_trip_reserved_ids_and_save_load(tmp_path: pytest.TempPathFactory) -> None:
    tokenizer = ByteBPETokenizer.train(["café café", "hello hello"], vocab_size=280)
    text = "UTF-8: café, 東京, 🙂"
    assert tokenizer.decode(tokenizer.encode(text)) == text
    assert [tokenizer.special_to_id[token] for token in RESERVED_TOKENS] == list(range(7))
    path = tmp_path / "tokenizer"
    tokenizer.save(path)
    loaded = ByteBPETokenizer.load(path)
    assert loaded.sha256 == tokenizer.sha256
    assert loaded.encode(text) == tokenizer.encode(text)

    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["reserved_tokens"]["<|bos|>"] = 99
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest"):
        ByteBPETokenizer.load(path)


def test_reference_tokenizer_trainer_fails_fast_above_safe_boundary() -> None:
    oversized = "x" * (REFERENCE_TRAINER_MAX_BYTES + 1)
    with pytest.raises(ValueError, match="reference BPE trainer"):
        ByteBPETokenizer.train([oversized], vocab_size=263)


def test_chat_mask_trains_only_assistant_content_and_end() -> None:
    tokenizer = ByteBPETokenizer()
    messages = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "4"},
    ]
    encoding = encode_conversation(messages, tokenizer)
    supervised = [
        token
        for token, active in zip(encoding.input_ids, encoding.assistant_mask, strict=True)
        if active
    ]
    assert supervised == [*tokenizer.encode("4"), tokenizer.special_to_id["<|end|>"]]
    assert render_conversation(messages).startswith("<|bos|><|system|>")
    labels = encoding.labels()
    assert sum(label != -100 for label in labels) == len(supervised)


def test_deterministic_shards_checksums_rank_partition_and_resume(tmp_path) -> None:
    tokenizer = ByteBPETokenizer()
    records = [
        FineWebRecord(str(index), f"document {index} " * 4, "revision") for index in range(30)
    ]
    output = tmp_path / "dataset"
    manifest = prepare_shards(records, tokenizer, output, shard_tokens=80, validation_fraction=0.2)
    assert manifest["token_count"] > 0
    assert {item["split"] for item in manifest["shards"]} == {"train", "validation"}

    rank_zero = ShardReader(output, split="train", rank=0, world_size=2)
    rank_one = ShardReader(output, split="train", rank=1, world_size=2)
    assert {item["path"] for item in rank_zero.shards}.isdisjoint(
        {item["path"] for item in rank_one.shards}
    )
    reader = ShardReader(output, split="train")
    stream = reader.iter_tokens()
    prefix = [next(stream) for _ in range(5)]
    cursor = prefix[-1][1]
    remainder = [token for token, _ in reader.iter_tokens(cursor)]
    complete = [token for token, _ in reader.iter_tokens(Cursor())]
    assert [token for token, _ in prefix] + remainder == complete
    with pytest.raises(ValueError, match="offset"):
        list(reader.iter_tokens(Cursor(0, -1)))
    with pytest.raises(ValueError, match="shard index"):
        list(reader.iter_tokens(Cursor(len(reader.shards) + 1, 0)))

    first = output / manifest["shards"][0]["path"]
    first.write_bytes(first.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        ShardReader(output, split=manifest["shards"][0]["split"])


def test_schema_allowlist_and_sft_padding_mask() -> None:
    record = ConversationRecord.from_mapping(
        {
            "id": "one",
            "messages": [
                {"role": "user", "content": "Question"},
                {"role": "assistant", "content": "Answer"},
            ],
            "source": "fixture",
            "license": "CC0-1.0",
        }
    )
    assert require_allowlisted([record], {("fixture", "CC0-1.0")}) == [record]
    with pytest.raises(ValueError, match="unapproved"):
        require_allowlisted([record], set())
    batch = collate_sft([record], ByteBPETokenizer(), max_seq_len=64)
    assert batch.loss_mask.any()
    assert bool((batch.targets[~batch.loss_mask] == -100).all())
    assert not batch.loss_mask[0, 0]

    no_assistant = ConversationRecord(
        id="prompt-only",
        messages=({"role": "user", "content": "Question"},),
        source="fixture",
        license="CC0-1.0",
    )
    with pytest.raises(ValueError, match="assistant targets"):
        collate_sft([no_assistant], ByteBPETokenizer(), max_seq_len=64)

    # Ignored prompt positions remain harmless even if a caller drops the
    # redundant explicit mask.
    assert torch.equal(batch.targets.eq(-100), ~batch.loss_mask)


def test_split_is_stable() -> None:
    assert assign_split("stable-id") == assign_split("stable-id")
