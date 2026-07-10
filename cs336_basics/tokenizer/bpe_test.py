'''Unit tests for BPE.

uv run pytest cs336_basics/tokenizer/bpe_test.py
'''

from cs336_basics.tokenizer.bpe import BytePairEncoder

def test_init():
    special_tokens = ['sp1', 'sp2', 'sp3']
    bpe = BytePairEncoder(
        pretokens = ['hello', ' world'],
        # anything less than 256 + num special tokens is fine.
        vocab_size = 0,
        special_tokens = special_tokens,
    )
    vocabs, merges = bpe.train()
    expected_vocabs = {
        i: bytes([i])
        for i in range(256)
    }
    expected_vocabs.update({
        256 + i: sp.encode()
        for i, sp in enumerate(special_tokens)
    })
    assert vocabs == expected_vocabs
    assert len(merges) == 0

def test_example():
    special_tokens = ['<|endoftext|>']
    pretokens = [
        'low',
        'low',
        'low',
        'low',
        'low',
        'lower',
        'lower',
        'wildest',
        'wildest',
        'wildest',
        'newest',
        'newest',
        'newest',
        'newest',
        'newest',
        'newest',
    ]
    bpe = BytePairEncoder(
        pretokens = pretokens,
        # anything less than 256 + num special tokens is fine.
        vocab_size = 263,
        special_tokens = special_tokens,
    )
    vocabs, merges = bpe.train()
    assert len(vocabs) == 263
    assert vocabs[256] == b'<|endoftext|>'
    assert vocabs[257] == b'st'
    assert vocabs[258] == b'est'
    assert vocabs[259] == b'ow'
    assert vocabs[260] == b'low'
    assert vocabs[261] == b'west'
    assert vocabs[262] == b'ne'
    assert merges == [
        (b's', b't'),
        (b'e', b'st'),
        (b'o', b'w'),
        (b'l', b'ow'),
        (b'w', b'est'),
        (b'n', b'e'),
    ]
