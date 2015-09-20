import numpy as np
from collections import defaultdict


class DataIterator(object):
    def __init__(self, tune_lens, batch_size, random_lens=False):
        self.batch_size = batch_size
        self.ntunes = len(tune_lens)

        self.len2idx = defaultdict(list)
        for i in xrange(self.ntunes):
            self.len2idx[tune_lens[i]].append(i)

        self.random_lens = random_lens
        self.rng = np.random.RandomState(42)

    def __iter__(self):
        if self.random_lens:
            while True:
                for batch_offsets in self.__iter_random_lens():
                    yield batch_offsets
        else:
            while True:
                for batch_offsets in self.__iter_homogeneous_lens():
                    yield batch_offsets

    def __iter_random_lens(self):
        available_idxs = np.arange(self.ntunes)
        while len(available_idxs) >= self.batch_size:
            rand_idx = self.rng.choice(range(len(available_idxs)), size=self.batch_size, replace=False)
            yield available_idxs[rand_idx]
            available_idxs = np.delete(available_idxs, rand_idx)

    def __iter_homogeneous_lens(self):
        for idxs in self.len2idx.itervalues():
            self.rng.shuffle(idxs)

        progress = defaultdict(int)
        available_lengths = self.len2idx.keys()

        batch_idxs = []
        b_size = self.batch_size

        get_tune_len = lambda: self.rng.choice(available_lengths)
        k = get_tune_len()

        while available_lengths:
            batch_idxs.extend(self.len2idx[k][progress[k]:progress[k]+b_size])
            progress[k] += b_size
            if len(batch_idxs) == self.batch_size:
                yield batch_idxs
                batch_idxs = []
                b_size = self.batch_size
                k = get_tune_len()
            else:
                b_size = self.batch_size - len(batch_idxs)
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