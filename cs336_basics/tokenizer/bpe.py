"""Implementation of BPE algorithm."""

import collections
import itertools

import tqdm

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
        # Keeps track of pretoken counts. NOTE: the values
        # in this counter NEVER changes, as they reflect the
        # frequency of pretokens in corpus. What does change
        # are the keys, which are tuple of bytes elements,
        # where each bytes element correspond to some vocab,
        # and as BPE evolves with merging, the freq's keys
        # may change to different tuple of bytes, but the
        # concatinated bytes together is ALWAYS the pretoken.
        self._freq: dict[tuple[bytes, ...], int] \
            = collections.Counter()
        for pt in self._pretokens:
            key = tuple([
                bytes([b])
                for b in pt.encode()
            ])
            self._freq[key] += 1

    def _pair_to_merge(self) -> tuple[bytes, bytes]:
        '''Finds the most common bytes pair to merge.
        
        On tie, return the lexically largest candidate.
        '''
        pairs_counter = collections.Counter()
        # reverse index to update freq
        pair_to_pts = collections.defaultdict(set)
        for pt, count in self._freq.items():
            for l, r in itertools.pairwise(pt):
                pairs_counter[(l, r)] += count
                pair_to_pts[(l, r)].add(pt)
        max_cnt = max(pairs_counter.values())
        candidates = sorted([
            pair
            for pair, cnt in pairs_counter.items()
            if cnt == max_cnt
        ], key=lambda pair: pair[0] + pair[1])
        candidate = candidates[-1]

        # Update freq. For each pretoken subject to update
        # (i.e. where the candidate is a substring) it runs a
        # pairwise sliding window over the pretoken, to merge
        # relevant vocab pairs.
        for pt in pair_to_pts[candidate]:
            pt_cnt = self._freq[pt]
            del self._freq[pt]
            new_pt = []
            i = 0
            while i < len(pt):
                if pt[i:i+2] == candidate:
                    new_pt.append(candidate[0] + candidate[1])
                    i += 2
                else:
                    new_pt.append(pt[i])
                    i += 1
            new_pt = tuple(new_pt)
            self._freq[new_pt] = pt_cnt
        return candidate

    def _merge(self) -> None:
        '''Exercises 1 round of BPE merging.'''
        l, r = self._pair_to_merge()
        new_vocab = l + r
        # Update merge history.
        self._merges.append((l, r))
        # Update vocab.
        self._vocabs[len(self._vocabs)] = new_vocab

    def train(self) -> \
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        '''Runs BPE training until accumulating enough vocabs.'''
        total_merges = self._target_vocab_size - len(self._vocabs)
        
        with tqdm.tqdm(total=total_merges, desc='BPE training') as pbar:
            while len(self._vocabs) < self._target_vocab_size:
                old_vocabs_count = len(self._vocabs)
                self._merge()
                new_vocabs_count = len(self._vocabs)
                pbar.update(new_vocabs_count - old_vocabs_count)
        return self._vocabs, self._merges
