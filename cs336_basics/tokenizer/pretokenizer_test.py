'''Unit tests for pretokenizer.

uv run pytest cs336_basics/tokenizer/pretokenizer_test.py
'''

from cs336_basics.tokenizer.pretokenizer import SpecialTokenAwarePretokenizer

def test_special_token_aware_pretokenizer():
    pretokenizer = SpecialTokenAwarePretokenizer(
        special_tokens = [
            '<sp1>',
            '<sp2>',
        ]
    )
    text = 'the<sp1>dog<sp2>ate'
    pretokens = pretokenizer.pretokenize(text)
    assert pretokens == [
        'the',
        '<sp1>',
        'dog',
        '<sp2>',
        'ate',
    ]