'''Optimized implementation of BPE.

The vanilla BPE implementation suffers from a performance hotspot issue where,
whenever we need to merge the most common pair, we brute-force update all
pretoken entries the in frequency counter, even though in practice only a
small subset of pretokens who has the merged pair needs update.

This optimization version addresses it by introducing clear API boundaries
between (a) the pretoken frequency counter, which maps from (pretoken, list
of token bytes sequences, count of pretoken), and (b) the byte pair counter,
which maps from each pretoken pair to their count. Communication is as follows:

1. (b) finds the most common (l, r) pair to merge.
2. (b) sends (l, r) to (a)
3. (a) updates internal counter, records all affected pairs.
4. (a) sends all affected pairs back to (b)
5. (b) updates its state.
'''


import collections
import heapq
import itertools

import tqdm

class PretokenCounter:
    def __init__(self):
        self._freq: dict[tuple[bytes, ...], int] \
            = collections.Counter()
        # Internal index to track mapping from bytes pair to
        # pretoken keys in self._freq.
        self._pair_to_freq_keys: dict[
            tuple[bytes, bytes],
            set[tuple[bytes, ...]],
        ] = collections.defaultdict(set)
        
    def update_freq_amount(self, key: tuple[bytes, ...], amount: int)\
        -> None:
        self._freq[key] += amount
        if self._freq[key] == amount:
            # Update internal index for first time.
            for l, r in itertools.pairwise(key):
                self._pair_to_freq_keys[(l, r)].add(key)

    def merge(self, byte_pair: tuple[bytes, bytes])\
        -> dict[tuple[bytes, bytes], int]:
        '''Merges the request bytes pair and update internal counter.
        
        Args:
            bytes_pair: bytes pair to merge.

        Returns:
            Deltas to apply to byte pair counter.
        '''
        pair_delta: dict[tuple[bytes, bytes], int] = collections.Counter()
        for pretoken in list(self._pair_to_freq_keys[byte_pair]):
            pretoken_count = self._freq.pop(pretoken)

            # Subtract old pretoken's count.
            for l, r in itertools.pairwise(pretoken):
                pair_delta[(l, r)] -= pretoken_count
                # Clean up pretoken in self._pair_to_freq_keys:
                if (l, r) in self._pair_to_freq_keys:
                    keys  = self._pair_to_freq_keys[(l, r)]
                    keys.discard(pretoken)
                    if not keys:
                        del self._pair_to_freq_keys[(l, r)]
            
            # Construct new pretoken bytes sequence.
            new_pretoken = []
            i = 0
            while i < len(pretoken):
                if pretoken[i:i+2] == byte_pair and i + 1 < len(pretoken):
                    new_pretoken.append(byte_pair[0] + byte_pair[1])
                    i += 2
                else:
                    new_pretoken.append(pretoken[i])
                    i += 1
            new_pretoken = tuple(new_pretoken)
            self._freq[new_pretoken] = pretoken_count

            # Add new pretoken's contribution.
            for l, r in itertools.pairwise(new_pretoken):
                pair_delta[(l, r)] += pretoken_count
                self._pair_to_freq_keys[(l, r)].add(new_pretoken)
        return pair_delta

class BytePairCounter:

    def __init__(self):
        # Counts frequency of each bytes pair. This is always kept most
        # up-to-date.
        self._pair_count = collections.Counter()

        # Max heap of bytes pair counter, sort key is
        # count + lexico order of the bytes pair. Content
        # is stale if count is inconsistent with self._pair_count above.
        self._max_heap = []

    def update_counts(
            self,
            pair_delta: dict[tuple[bytes, bytes], int]) -> None:
        for pair, delta in pair_delta.items():
            self._pair_count[pair] += delta
            if self._pair_count[pair] <= 0:
                del self._pair_count[pair]
            else:
                heapq.heappush(
                    self._max_heap,
                    (
                        -self._pair_count[pair],
                        # Rank by max lexico ordering.
                        tuple(-b for b in (pair[0] + pair[1])),
                        pair,
                    )
                )

    def find_most_common_byte_pair(self)\
        -> tuple[bytes, bytes]:
        while self._max_heap:
            neg_count, _neg_key, pair = heapq.heappop(self._max_heap)
            if self._pair_count[pair] != -neg_count:
                # Stale entry.
                continue
            # Note: no need to push back again, because most common bytes
            # will be evicted and merged later anyway.
            # heapq.heappush(self._max_heap, bytes_seq, pair)
            return pair
        raise RuntimeError('Unexpectedly reached end of queue.')

class BytePairEncoder:
    """Bype pair encoder."""

    def __init__(
        self, 
        vocab_size: int, 
        special_tokens: list[str],
        pretokens: list[str] | None = None,
    ):
        # Holds original init input.
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
        
        self._pretoken_counter = PretokenCounter()
        self._byte_pair_counter = BytePairCounter()
        
        if pretokens is not None:
            self.update_pretokens(pretokens)
        
    def update_pretokens(self, pretokens: list[str]) -> None:
        '''Updates internal freq counter for new pretokens.'''
        pair_delta = collections.Counter()
        for pt in pretokens:
            key = tuple([
                bytes([b])
                for b in pt.encode()
            ])
            self._pretoken_counter.update_freq_amount(key, 1)
            for l, r in itertools.pairwise(key):
                pair_delta[(l, r)] += 1
        self._byte_pair_counter.update_counts(pair_delta)

    def _pair_to_merge(self) -> tuple[bytes, bytes]:
        '''Finds the most common bytes pair to merge.
        
        On tie, return the lexically largest candidate.
        '''
        candidate: tuple[bytes, bytes] =\
            self._byte_pair_counter.find_most_common_byte_pair()
        pair_delta =\
            self._pretoken_counter.merge(candidate)
        self._byte_pair_counter.update_counts(pair_delta)
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
