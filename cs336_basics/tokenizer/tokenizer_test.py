'''Unit tests for Tokenizer.

uv run pytest cs336_basics/tokenizer/tokenizer_test.py
'''

from cs336_basics.tokenizer.tokenizer import Tokenizer

def test_encode_ascii():
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
    expected_token_ids = [9, 7, 1, 5, 10, 3]
    assert tokenizer.encode('the cat ate') == expected_token_ids
    assert list(tokenizer.encode_iterable('the cat ate')) == expected_token_ids

def test_encode_non_ascii():
    vocabs = {
        i: bytes([i])
        for i in range(256)
    }
    vocabs[256] = '👋'.encode()[:2]
    vocabs[257] = '👋'.encode()[2:]
    merges = [
        (bytes([vocabs[256][0]]), bytes([vocabs[256][1]])),
        (bytes([vocabs[257][0]]), bytes([vocabs[257][1]])),
    ]
    tokenizer = Tokenizer(
        vocabs=vocabs,
        merges=merges,
    )
    text = '👋'
    expected_ids = [256, 257]
    actual_ids = tokenizer.encode(text)
    assert actual_ids == expected_ids

def test_encode_look_back():
    vocabs = {
        0: 'a'.encode(),
        1: 'b'.encode(),
        2: 'c'.encode(),
        3: 'd'.encode(),
        4: 'cd'.encode(),
        5: 'bcd'.encode(),
    }
    merges = [
        ('c'.encode(), 'd'.encode()),
        ('b'.encode(), 'cd'.encode()),
    ]
    text = 'abcd'
    tokenizer = Tokenizer(vocabs=vocabs, merges=merges)
    assert tokenizer.encode(text) == [0, 5]

def test_decode_non_ascii():
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
