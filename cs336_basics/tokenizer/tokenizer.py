from collections import deque
from functools import cache
from typing import Iterable, Iterator

from cs336_basics.tokenizer.pretokenizer import SpecialTokenAwarePretokenizer

# From https://github.com/openai/gpt-2/blob/master/src/encoder.py#L53C31-L53C112
PAT = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    '''BPE tokenizer.'''

    def __init__(
        self,
        vocabs: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None):
        
        # Holds original init inputs.
        self.vocabs = vocabs
        self.merges = merges
        self.special_tokens = special_tokens

        # Derived members.
        self.vocabs_to_index = {
            v: i
            for i, v in vocabs.items()
        }
        self.pretokenizer = SpecialTokenAwarePretokenizer(
            special_tokens=special_tokens,
        )

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None):
        ...

    def encode(self, text: str) -> list[int]:
        pretokens = self.pretokenizer.pretokenize(text)

        @cache
        def encode_pretoken(pretoken: str) -> list[int]:
            pretoken = pretoken.encode()
            if pretoken in self.vocabs_to_index:
                return [self.vocabs_to_index[pretoken]]
            pretoken: deque[bytes] = deque([
                bytes([i])
                for i in pretoken
            ])
            while len(pretoken) >= 2:
                first, second = pretoken.popleft(), pretoken.popleft()
                found_merge = False
                for merge in self.merges:
                    assert len(merge) == 2
                    if (first, second) == merge:
                        pretoken.appendleft(first + second)
                        found_merge = True
                        break
                if not found_merge:
                    # No more merges combine active list of pretokens' stop.
                    pretoken.appendleft(second)
                    pretoken.appendleft(first)
                    break
            return [
                self.vocabs_to_index[v]
                for v in pretoken
            ]

        ids = []
        for pretoken in pretokens:
            ids.extend(encode_pretoken(pretoken))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) \
        -> Iterator[int]:
        ...

    def decode(self, ids: list[int]) -> str:
        vocabs = [self.vocabs[i] for i in ids]
        return b''.join(vocabs).decode(errors='replace')
