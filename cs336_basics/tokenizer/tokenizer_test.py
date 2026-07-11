'''Unit tests for Tokenizer.

uv run pytest cs336_basics/tokenizer/tokenizer_test.py
'''

from cs336_basics.tokenizer.tokenizer import Tokenizer

def test_encode():
    ...

def test_encode_iterable():
    ...

def test_decode():
    tokenizer = Tokenizer(
        vocabs = {
            0: 'hello'.encode(),
            1: '你好'.encode(),
            2: '👋'.encode(),
            3: b'\xc2',  # Invalid UTF8 bytes
        },
        merges = [],
        special_tokens = [],
    )

    assert tokenizer.decode([1, 2, 0]) == '你好👋hello'
    assert tokenizer.decode([1, 2, 0, 3]) == '你好👋hello�'
