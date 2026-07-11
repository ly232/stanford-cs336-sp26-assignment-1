r'''Special-token aware pretokenizer.

This is a simple decorator on top of gpt2's pretokenization step,
where the pretokenization is made special-token-aware in the following way:
* Special tokens are NEVER pretokenized by regex (effectively ignored).
* Special tokens' positions are preserved in the output of pretokenization.

Example:
  input: 'a<sp1>b<sp2>c'
  output: ['a_pt1', '<sp1>', 'b_pt1'. 'b_pt2', '<sp2>', 'c_pt1']
'''

import regex

# From https://github.com/openai/gpt-2/blob/master/src/encoder.py#L53C31-L53C112
PAT = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class SpecialTokenAwarePretokenizer:
    def __init__(
        self,
        special_tokens: list[str] | None = None,
        pretokenization_regex: str = PAT,
    ):
        self.special_tokens = special_tokens if special_tokens else []
        self.pretokenization_regex = pretokenization_regex

    def pretokenize(self, text) -> list[str]:
        def _pretokenize_internal(
                text: str,
                special_tokens: list[str] | None,
            ) -> list[str]:
            if not special_tokens:
                return regex.findall(self.pretokenization_regex, text)
            sp = special_tokens[0]
            chunks = text.split(sp)
            pretokenized_chunks: list[str] = []
            for chunk in chunks:
                pretokenized_chunks.extend(
                    _pretokenize_internal(chunk, special_tokens[1:]))
                pretokenized_chunks.append(sp)
            pretokenized_chunks.pop()  # pop off the last special token
            return pretokenized_chunks
        return _pretokenize_internal(text, self.special_tokens)
