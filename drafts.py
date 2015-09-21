# class TrainIterator():
#     def __init__(self, data, batch_size, seq_length):
#         self.data = data
#         self.batch_size = batch_size
#         self.seq_length = seq_length
#         self.chars_per_batch = batch_size * seq_length
#         self.idx = 0
#         self.data_length = len(data)
#
#     def __iter__(self):
#         return self
#
#     def next(self):
#         next_idx = self.idx + self.chars_per_batch
#         if next_idx + 1 < self.data_length:
#             x = data[self.idx:next_idx]
#             y = np.append(x[1:], data[next_idx + 1])
#
#             x = np.reshape(x, (self.batch_size, self.seq_length))
#             y = np.reshape(y, (self.batch_size, self.seq_length))
#             self.idx = next_idx
#             return x, y
#         else:
#             self.idx = 0
#             raise StopIteration()

import theano
import lasagne
import numpy as np
import theano.tensor as T
from lasagne.layers import *
import time
from itertools import izip

batch_size = 2
emb_size, l1_size, l2_size, l3_size = 16, 8, 8, 8

with open('input_test.txt', 'r') as f:
    data = f.read()

idx2token = list(set(data.split()))
vocab_size = len(idx2token)
token2idx = dict(izip(idx2token, xrange(vocab_size)))
tunes = data.split('\n\n')
del data

for i in sorted(idx2token):
    print i
print vocab_size


tunes.sort(key=lambda x: len(x), reverse=True)
ntunes = len(tunes)
print 'n tunes: ', ntunes

tunes = [[token2idx[c] for c in t.split()] for t in tunes]
lens = np.array([len(t) for t in tunes])
offsets = np.concatenate(([0], np.cumsum(lens)))
max_len = max(lens)
print tunes[0]
print tunes[2]

tunes = np.array([i for tune in tunes for i in tune], dtype='float32')
print '==============='

print 'Load data to GPU'

x_shared = theano.shared(tunes)
x_batch_shared = theano.shared(np.zeros((batch_size, max_len), dtype='float32'))
mask_shared = theano.shared(np.zeros((batch_size, max_len), dtype='float32'))

x_s = T.matrix('inputs', dtype='float32')
mask_s = T.matrix('mask', dtype='float32')
y_s = T.matrix('targets', dtype='float32')

l_inp = InputLayer((batch_size, None), input_var=T.cast(x_s, 'int32'))
l_emb = EmbeddingLayer(l_inp, input_size=vocab_size, output_size=emb_size)
output = get_output(l_emb, T.cast(x_s, 'int32'))


idx = np.array([0, 2])
batch_max_len = max(lens[idx])
for b, i in enumerate(idx):
 x_batch_shared = T.set_subtensor(x_batch_shared[b, 0:lens[i]], x_shared[offsets[i]:offsets[i+1]])
 mask_shared = T.set_subtensor(x_batch_shared[b, 0:lens[i]-1], 1)

iter_train = theano.function([], output, givens={x_s: x_batch_shared})
print iter_train()
