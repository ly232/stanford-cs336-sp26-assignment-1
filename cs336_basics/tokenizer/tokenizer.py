from typing import Iterable, Iterator

class Tokenizer:
    '''BPE tokenizer.'''

    def __init__(
        self,
        vocabs: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str]):
        
        # Holds original init inputs.
        self.vocabs = vocabs
        self.merges = merges
        self.special_tokens = special_tokens

    def encode(self, text: str) -> list[int]:
        ...

    def encode_iterable(self, iterable: Iterable[str]) \
        -> Iterator[int]:
        ...

    def decode(self, ids: list[int]) -> str:
        vocabs = [self.vocabs[i] for i in ids]
        return b''.join(vocabs).decode(errors='replace')
