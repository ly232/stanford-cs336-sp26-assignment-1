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
        raise NotImplementedError

    def encode_iterable(self, iterable: Iterable[str]) \
        -> Iterator[int]:
        pretokens = self.pretokenizer.pretokenize(iterable)

        @cache
        def encode_pretoken(pretoken: str) -> list[int]:
            pretoken = pretoken.encode()
            if pretoken in self.vocabs_to_index:
                return [self.vocabs_to_index[pretoken]]
            bytes_seq: deque[bytes] = deque([
                bytes([i])
                for i in pretoken
            ])
            def maybe_merge(bytes_seq: deque[bytes]) -> bool:
                '''One pass through of bytes seq to merge.
                
                Returns true iff merge happened.
                '''
                if len(bytes_seq) < 2:
                    return False

                buffer: deque[bytes] = deque([])
                # Nesting order matters. Must iterate merges first, then
                # bytes_seq, in order to favor earlier merges.
                for merge in self.merges:
                    while len(bytes_seq) >= 2:
                        l, r = bytes_seq.popleft(), bytes_seq.popleft()
                        buffer.append(l)
                        buffer.append(r)
                        if (l, r) == merge:
                            buffer.pop()
                            buffer.pop()
                            buffer.append(l + r)
                            while buffer:
                                bytes_seq.appendleft(buffer.pop())
                            return True
                        # we want pairwise sliding window, not tumbling window.
                        bytes_seq.appendleft(buffer.pop())
                    while buffer:
                        bytes_seq.appendleft(buffer.pop())
                return False
            while maybe_merge(bytes_seq):
                pass
            return [
                self.vocabs_to_index[v]
                for v in bytes_seq
            ]

        for pretoken in pretokens:
            ids = encode_pretoken(pretoken)
            for id in ids:
                yield id

    def encode(self, text: str) -> list[int]:
        return [id for id in self.encode_iterable(text)]

    def decode(self, ids: list[int]) -> str:
        vocabs = [self.vocabs[i] for i in ids]
        return b''.join(vocabs).decode(errors='replace')
