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
