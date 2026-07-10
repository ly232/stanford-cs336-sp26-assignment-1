"""Implementation of BPE algorithm."""

import collections
import itertools

class BytePairEncoder:
    """Bype pair encoder."""

    def __init__(
        self, 
        pretokens: list[str], 
        vocab_size: int, 
        special_tokens: list[str]):
        
        # Holds original init input.
        self._pretokens = pretokens
        self._target_vocab_size = vocab_size
        self._special_tokens = special_tokens

        # State variables.
        self._vocabs: dict[int, bytes] = {
            i: bytes([i])
            for i in range(256)
        }
        for sp in special_tokens:
            self._vocabs[len(self._vocabs)] = sp.encode()
        self._merges: list[tuple[bytes, bytes]] = []
        self._freq: dict[tuple[bytes, bytes], int] \
            = collections.Counter()
        for pt in self._pretokens:
            for l, r in itertools.pairwise(pt.encode()):
                self._freq[(l, r)] += 1

    def _merge(self) -> None:
        '''Exercises 1 round of BPE merging.'''
        ...

    def train(self) -> \
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        '''Runs BPE training until accumulating enough vocabs.'''
        while len(self._vocabs) < self._target_vocab_size:
            self._merge()
        return self._vocabs, self._merges
