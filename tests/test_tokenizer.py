from corvusai.tokenization.byte_tokenizer import decode, encode


def test_utf8_roundtrip():
    value = "def café():\n    return '🦅'\n"
    assert decode(encode(value)) == value
