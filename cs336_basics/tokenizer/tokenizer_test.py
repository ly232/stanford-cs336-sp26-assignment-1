'''Unit tests for Tokenizer.

uv run pytest cs336_basics/tokenizer/tokenizer_test.py
'''

from cs336_basics.tokenizer.tokenizer import Tokenizer

def test_encode():
    tokenizer = Tokenizer(
        vocabs = {
            0: b' ',
            1: b'a',
            2: b'c',
            3: b'e',
            4: b'h',
            5: b't',
            6: b'th',
            7: b' c',
            8: b' a',
            9: b'the',
            10: b' at',
        },
        merges = [
            (b't', b'h'),
            (b' ', b'c'),
            (b' ', b'a'),
            (b'th', b'e'),
            (b' a', b't'),
        ],
    )
    ids = tokenizer.encode('the cat ate')
    print(ids)
    assert ids == [9, 7, 1, 5, 10, 3]

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
    )

    assert tokenizer.decode([1, 2, 0]) == '你好👋hello'
    assert tokenizer.decode([1, 2, 0, 3]) == '你好👋hello�'
