from __future__ import annotations

import pytest
import torch

from nanogpt.config import SourceConfig
from nanogpt.data.chat import collate_sft, encode_conversation, render_conversation
from nanogpt.data.schemas import ConversationRecord, FineWebRecord
from nanogpt.data.shards import TokenLoader, prepare_shards
from nanogpt.data.sources import (
    arithmetic_conversations,
    assign_split,
    conversations,
    gsm8k_partition,
    require_allowlisted,
)
from nanogpt.tokenizer import ByteBPETokenizer


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
    assert sum(label != -100 for label in encoding.labels()) == len(supervised)


def test_sft_collation_pads_and_ignores_non_assistant_targets() -> None:
    tokenizer = ByteBPETokenizer()
    records = list(arithmetic_conversations(3))
    batch = collate_sft(records, tokenizer, max_seq_len=128)
    assert batch.input_ids.shape == batch.targets.shape
    supervised = batch.targets.ne(-100)
    expected = sum(
        sum(encode_conversation(record.messages, tokenizer).assistant_mask) for record in records
    )
    assert int(supervised.sum()) == expected
    assert not supervised[:, 0].any()  # BOS -> role marker is never a target.
    with pytest.raises(ValueError, match="max_seq_len"):
        collate_sft(records, tokenizer, max_seq_len=4)
    prompt_only = ConversationRecord(
        "prompt-only", ({"role": "user", "content": "Question"},), "fixture", "CC0-1.0"
    )
    with pytest.raises(ValueError, match="assistant targets"):
        collate_sft([prompt_only], tokenizer, max_seq_len=128)


def _prepare(tmp_path, *, shard_tokens: int = 80) -> tuple[object, dict[str, object]]:
    records = [FineWebRecord(str(i), f"document {i} " * 4, "revision") for i in range(40)]
    output = tmp_path / "dataset"
    manifest = prepare_shards(
        records, ByteBPETokenizer(), output, shard_tokens=shard_tokens, validation_fraction=0.2
    )
    return output, manifest


def test_shards_split_by_document_and_refuse_overwrite(tmp_path) -> None:
    _, manifest = _prepare(tmp_path)
    assert {item["split"] for item in manifest["shards"]} == {"train", "validation"}  # type: ignore[index]
    with pytest.raises(FileExistsError):
        _prepare(tmp_path)


def test_token_loader_resumes_exactly_and_wraps_epochs(tmp_path) -> None:
    output, _ = _prepare(tmp_path)
    loader = TokenLoader(output, split="train", batch_size=2, seq_len=16)
    reference = [loader.next_batch() for _ in range(60)]  # Enough to wrap an epoch.
    assert loader.cursor.epoch >= 1

    resumed = TokenLoader(output, split="train", batch_size=2, seq_len=16)
    for _ in range(5):
        resumed.next_batch()
    state = resumed.state_dict()
    fresh = TokenLoader(output, split="train", batch_size=2, seq_len=16)
    fresh.load_state_dict(state)
    for expected in reference[5:]:
        actual = fresh.next_batch()
        assert torch.equal(actual.input_ids, expected.input_ids)
        assert torch.equal(actual.targets, expected.targets)
    # Targets are inputs shifted by one token.
    assert torch.equal(reference[0].input_ids[0, 1:], reference[0].targets[0, :-1])


def test_token_loader_ranks_are_disjoint_and_checksums_verified(tmp_path) -> None:
    output, manifest = _prepare(tmp_path, shard_tokens=40)
    rank_zero = TokenLoader(output, split="train", batch_size=1, seq_len=4, rank=0, world_size=2)
    rank_one = TokenLoader(output, split="train", batch_size=1, seq_len=4, rank=1, world_size=2)
    assert {item["path"] for item in rank_zero.shards}.isdisjoint(
        {item["path"] for item in rank_one.shards}
    )
    first = output / manifest["shards"][0]["path"]  # type: ignore[index, operator]
    first.write_bytes(first.read_bytes() + b"xx")
    with pytest.raises(ValueError, match="checksum"):
        TokenLoader(output, split=manifest["shards"][0]["split"], batch_size=1, seq_len=4)  # type: ignore[index]


def test_allowlist_splits_and_source_mixture() -> None:
    record = next(arithmetic_conversations(1))
    assert require_allowlisted([record], {(record.source, record.license)}) == [record]
    with pytest.raises(ValueError, match="unapproved"):
        require_allowlisted([record], set())
    assert assign_split("stable-id") == assign_split("stable-id")
    assert {gsm8k_partition(f"gsm8k-train-{i}") for i in range(50)} == {"sft", "grpo"}
    mixture = conversations([SourceConfig(kind="arithmetic", examples=7, seed=3)])
    assert len(mixture) == 7
