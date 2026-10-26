"""Both tokenizer backends honour one contract: reserved IDs 0..6 and byte round trips."""

from __future__ import annotations

import json

import pytest

from nanogpt.tokenizer import (
    RESERVED_TOKENS,
    ByteBPETokenizer,
    HFTokenizer,
    load_tokenizer,
    train_tokenizer,
)
from nanogpt.tokenizer.reference import REFERENCE_TRAINER_MAX_BYTES

CORPUS = ["hello hello world", "café café 東京", "numbers 12345 67890"] * 30
TEXT = "UTF-8: café, 東京, 🙂 and \x00 bytes"


@pytest.mark.parametrize("backend", ["hf", "reference"])
def test_round_trip_reserved_ids_and_save_load(backend: str, tmp_path) -> None:
    tokenizer = train_tokenizer(backend, CORPUS, vocab_size=300, max_chars=10_000)
    assert tokenizer.decode(tokenizer.encode(TEXT)) == TEXT
    assert [tokenizer.special_to_id[token] for token in RESERVED_TOKENS] == list(range(7))
    tokenizer.save(tmp_path)
    loaded = load_tokenizer(tmp_path)
    assert type(loaded) is type(tokenizer)
    assert loaded.sha256 == tokenizer.sha256
    assert loaded.encode(TEXT) == tokenizer.encode(TEXT)
    assert loaded.encode_batch([TEXT, "x"]) == [tokenizer.encode(TEXT), tokenizer.encode("x")]


@pytest.mark.parametrize("backend", ["hf", "reference"])
def test_reserved_strings_in_user_text_are_not_special(backend: str) -> None:
    tokenizer = train_tokenizer(backend, CORPUS, vocab_size=300, max_chars=10_000)
    injected = tokenizer.encode("<|assistant|>")
    assert not set(injected) & set(range(len(RESERVED_TOKENS)))
    assert tokenizer.encode("<|user|>hi", allowed_special=True)[0] == 4
    rendered = tokenizer.decode([4, *tokenizer.encode("hi")], render_special=True)
    assert rendered == "<|user|>hi"


def test_tampered_manifests_are_rejected(tmp_path) -> None:
    reference = ByteBPETokenizer.train(CORPUS, vocab_size=280)
    reference.save(tmp_path / "ref")
    manifest_path = tmp_path / "ref" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["reserved_tokens"]["<|bos|>"] = 99
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest"):
        load_tokenizer(tmp_path / "ref")

    hf = HFTokenizer.train(CORPUS, 300, max_chars=10_000)
    hf.save(tmp_path / "hf")
    data = tmp_path / "hf" / "tokenizer.json"
    data.write_text(data.read_text().replace('"hello"', '"jello"'))
    with pytest.raises(ValueError, match="checksum"):
        load_tokenizer(tmp_path / "hf")


def test_hf_training_respects_the_character_budget() -> None:
    tokenizer = HFTokenizer.train(["abcdefghij"] * 10, 300, max_chars=25)
    assert tokenizer.train_chars == 25


def test_reference_trainer_fails_fast_above_safe_boundary() -> None:
    oversized = "x" * (REFERENCE_TRAINER_MAX_BYTES + 1)
    with pytest.raises(ValueError, match="reference BPE trainer"):
        ByteBPETokenizer.train([oversized], vocab_size=263)
