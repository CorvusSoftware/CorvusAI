from pathlib import Path

import pytest

from corvusai.tokenization.bpe import ByteBPE
from corvusai.tokenization.benchmark import benchmark


def test_bpe_is_deterministic_and_roundtrips(tmp_path: Path):
    documents = [b"def foo():\n    return 42\n" * 8, "café 🦅".encode("utf-8") * 4]
    first = ByteBPE.train(documents, merges=30)
    second = ByteBPE.train(documents, merges=30)
    assert first.merges == second.merges
    for document in documents + [bytes(range(256))]:
        assert first.decode_bytes(first.encode_bytes(document)) == document
    path = tmp_path / "tokenizer.json"
    first.save(path)
    restored = ByteBPE.load(path)
    assert restored.merges == first.merges
    assert benchmark(restored, documents)["bytes_per_token"] >= 1


def test_invalid_merges():
    with pytest.raises(ValueError):
        ByteBPE([(300, 1)])
    with pytest.raises(ValueError):
        ByteBPE.train([b"abc"], merges=-1)
