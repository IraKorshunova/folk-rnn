import numpy as np
from collections import defaultdict
from itertools import izip


class TrainIterator(object):
    def __init__(self, offsets, tune_lens, batch_size, random_lens=False):
        self.batch_size = batch_size
        self.offsets = offsets

        self.len2offsets = defaultdict(list)
        for k, v in izip(tune_lens, offsets):
            self.len2offsets[k].append(v)

        self.random_lens = random_lens
        self.rng = np.random.RandomState(42)

    def __iter__(self):
        while True:
            if self.random_lens:
                for batch_offsets in self.__iter_random_lens():
                    yield batch_offsets
            else:
                for batch_offsets in self.__iter_homogeneous_lens():
                    yield batch_offsets

    def __iter_random_lens(self):
        available_offsets = np.copy(self.offsets)
        while len(available_offsets) > self.batch_size:
            batch_offsets_idxs = self.rng.choice(range(len(available_offsets)), size=self.batch_size, replace=False)
            batch_offsets = available_offsets[batch_offsets_idxs]
            yield batch_offsets
            available_offsets = np.delete(available_offsets, batch_offsets_idxs)

    def __iter_homogeneous_lens(self):
        for offsets in self.len2offsets.itervalues():
            self.rng.shuffle(offsets)

        progress = defaultdict(int)
        available_lengths = self.len2offsets.keys()

        batch_offsets = []
        b_size = self.batch_size

        get_tune_len = lambda: self.rng.choice(available_lengths)
        k = get_tune_len()

        while available_lengths:
            batch_offsets.extend(self.len2offsets[k][progress[k]:progress[k]+b_size])
            progress[k] += b_size
            if len(batch_offsets) == self.batch_size:
                yield batch_offsets
                batch_offsets = []
                b_size = self.batch_size
                k = get_tune_len()
            else:
                b_size = self.batch_size - len(batch_offsets)
                i = available_lengths.index(k)
                del available_lengths[i]
                if not available_lengths:
                    break
                if i == 0:
                    k = available_lengths[0]
                elif i >= len(available_lengths) - 1:
                    k = available_lengths[-1]
                else:
                    k = available_lengths[i + self.rng.choice([-1, 0])]
        if batch_offsets:
            yield batch_offsets[:self.batch_size]


