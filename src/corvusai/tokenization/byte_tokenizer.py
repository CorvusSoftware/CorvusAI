"""Reversible UTF-8 byte tokenizer; no pretrained vocabulary."""


def encode(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def decode(tokens: list[int]) -> str:
    return bytes(tokens).decode("utf-8")
