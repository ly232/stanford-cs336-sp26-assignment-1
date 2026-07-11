from typing import Iterable, Iterator

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

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None):
        ...

    def encode(self, text: str) -> list[int]:
        # for sp in self.special_tokens:
        #     text = text.replace(sp, '')
        chunks = [(text, None)]
        for sp in self.special_tokens:
            new_chunks = []
            for chunk in chunks:
                chunk_split = chunk.split(sp)
                chunk_split = [(subtext, sp) for subtext in chunk_split]
                new_chunks.extend(chunk_split)
            chunks = new_chunks
        pretokens = regex.findall(PAT, text)

    def encode_iterable(self, iterable: Iterable[str]) \
        -> Iterator[int]:
        ...

    def decode(self, ids: list[int]) -> str:
        vocabs = [self.vocabs[i] for i in ids]
        return b''.join(vocabs).decode(errors='replace')
