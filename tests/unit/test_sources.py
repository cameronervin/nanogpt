"""Offline tests for the Hugging Face source adapters (no network)."""

from __future__ import annotations

from typing import Any

import pytest

from nanogpt.config import SourceConfig
from nanogpt.data import sources
from nanogpt.data.prepare import fit_conversation
from nanogpt.data.schemas import ConversationRecord, FineWebRecord
from nanogpt.data.shards import prepare_shards
from nanogpt.tokenizer import ByteBPETokenizer


def test_gsm8k_keeps_reasoning_and_strips_calculator_annotations() -> None:
    row = {
        "question": " Natalia sold 48 clips in April and half as many in May. ",
        "answer": "In May she sold 48/2 = <<48/2=24>>24 clips.\nIn total 48+24 = <<48+24=72>>72.\n"
        "#### 1,072",
    }
    problem = sources.parse_gsm8k(5, row)
    assert problem.id == "gsm8k-train-5"
    assert problem.question.startswith("Natalia") and problem.question.endswith("May.")
    assert problem.solution == "In May she sold 48/2 = 24 clips.\nIn total 48+24 = 72."
    assert problem.answer == "1072"
    with pytest.raises(ValueError, match="no '#### answer'"):
        sources.parse_gsm8k(0, {"question": "q", "answer": "no marker"})


def test_gsm8k_sft_targets_reasoning_then_final_answer(monkeypatch) -> None:
    rows = [{"question": f"Q{i}?", "answer": f"Step {i}.\n#### {i}"} for i in range(40)]
    monkeypatch.setattr(sources, "_load_dataset", lambda *a, **k: rows)
    source = SourceConfig(kind="gsm8k", examples=100)
    sft = sources.conversations([source])
    grpo = sources.prompts([source])
    assert sft and grpo
    assert {r.id for r in sft}.isdisjoint({p.id for p in grpo})
    reply = sft[0].messages[1]["content"]
    assert reply.startswith("Step ") and reply.endswith("</final>")
    assert all(p.answer.isdigit() for p in grpo)


def test_smoltalk_interleaves_subsets_evenly(monkeypatch) -> None:
    def fake_rows(source: SourceConfig, subset: str) -> list[dict[str, Any]]:
        message = [{"role": "user", "content": subset}, {"role": "assistant", "content": "ok"}]
        return [{"messages": message} for _ in range(10 if subset == "a" else 1)]

    monkeypatch.setattr(sources, "_smoltalk_rows", fake_rows)
    source = SourceConfig(kind="smoltalk", examples=5, subsets=["a", "b"], license="apache-2.0")
    records = sources.conversations([source])
    assert sorted(r.source.rsplit("/", 1)[1] for r in records) == ["a", "a", "a", "a", "b"]
    with pytest.raises(ValueError, match="license"):
        sources.conversations([SourceConfig(kind="smoltalk", examples=1, subsets=["a"])])


def test_long_chats_keep_whole_turn_prefix_ending_on_assistant() -> None:
    tokenizer = ByteBPETokenizer()
    messages = (
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "x" * 200},
        {"role": "assistant", "content": "y" * 200},
    )
    record = ConversationRecord("c", messages, "fixture", "CC0-1.0")
    assert fit_conversation(record, tokenizer, 1000) is record
    shortened = fit_conversation(record, tokenizer, 50)
    assert shortened is not None and shortened.messages == messages[:2]
    assert fit_conversation(record, tokenizer, 5) is None


def test_max_tokens_stops_at_the_first_document_past_the_budget(tmp_path) -> None:
    records = [FineWebRecord(str(i), "word " * 50, "rev") for i in range(100)]
    manifest = prepare_shards(
        records, ByteBPETokenizer(), tmp_path / "d", validation_fraction=0.0, max_tokens=1000
    )
    per_document = 252  # bos + 250 bytes + eos
    assert manifest["documents"] == 4  # 3 documents = 756 < 1000; the 4th crosses it
    assert manifest["token_count"] == 4 * per_document
