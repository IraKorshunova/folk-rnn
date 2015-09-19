import theano
import lasagne
import numpy as np
import theano.tensor as T
from lasagne.layers import *
from itertools import izip
from train_iter import TrainIterator


start_symbol, end_symbol = '<s>', '</s>'
batch_size = 3
emb_size, l1_size, l2_size, l3_size = 16, 8, 8, 8

with open('input_test.txt', 'r') as f:
    data = f.read()

tokens_set = set(data.split())
tokens_set.update({start_symbol, end_symbol})

idx2token = list(tokens_set)
vocab_size = len(idx2token)
token2idx = dict(izip(idx2token, xrange(vocab_size)))
tunes = data.split('\n\n')
del data

# for i in sorted(idx2token):
# print i


tunes.sort(key=lambda x: len(x), reverse=True)
ntunes = len(tunes)

print 'vocab size', vocab_size
print 'n tunes: ', ntunes

tunes = [[token2idx[c] for c in [start_symbol] + t.split() + [end_symbol]] for t in tunes]
tune_lens = np.array([len(t) for t in tunes])
offsets = np.concatenate(([0], np.cumsum(tune_lens)[:-1]))
max_len = max(tune_lens)

print offsets
print tune_lens


iter = TrainIterator(offsets, tune_lens, batch_size, random_lens=False)

for i in iter:
    print i

exit(0)

# print [idx2token[c] for c in tunes[0]]
# print [idx2token[c] for c in tunes[2]]
# print [idx2token[c] for c in tunes[4]]

tunes = np.array([i for tune in tunes for i in tune], dtype='float32')

print 'Load data to GPU'
data_shared = theano.shared(tunes)
batch_shared = theano.shared(np.zeros((batch_size, max_len), dtype='float32'))
mask_shared = theano.shared(np.zeros((batch_size, max_len - 1), dtype='float32'))
tune_lens_shared = theano.shared(tune_lens)
offsets_shared = theano.shared(offsets)

idxs = T.ivector('idxs')

tune_lens_int = T.cast(tune_lens_shared, 'int32')
offsets_int = T.cast(offsets_shared, 'int32')

for i in xrange(batch_size):
    j = idxs[i]
    batch_shared = T.set_subtensor(batch_shared[i, 0:tune_lens_int[j]],
                                   data_shared[offsets_int[j]:offsets_int[j + 1]])
    mask_shared = T.set_subtensor(mask_shared[i, 0:tune_lens_int[j] - 1], 1)

max_seqlen = T.max(tune_lens_int[idxs])

x = T.cast(batch_shared[:, :max_seqlen - 1], 'int32')
y = T.cast(T.flatten(batch_shared[:, 1:max_seqlen]), 'int32')

mask_flat = T.flatten(mask_shared)

l_inp = InputLayer((batch_size, None), input_var=x)
l_emb = EmbeddingLayer(l_inp, input_size=vocab_size, output_size=emb_size)

l_mask = InputLayer(shape=(batch_size, None), input_var=mask_shared)

l1_lstm = LSTMLayer(l_emb, num_units=l1_size, peepholes=False, mask_input=l_mask)
l2_lstm = LSTMLayer(l1_lstm, num_units=l2_size, peepholes=False, mask_input=l_mask)
l3_lstm = LSTMLayer(l2_lstm, num_units=l3_size, peepholes=False, mask_input=l_mask)

l_reshp = ReshapeLayer(l3_lstm, (-1, l3_size))
l_out = DenseLayer(l_reshp, num_units=vocab_size, nonlinearity=T.nnet.softmax)
predictions = lasagne.layers.get_output(l_out)

print "Build model"
all_layers = lasagne.layers.get_all_layers(l_out)
all_params = lasagne.layers.get_all_params(l_out)
num_params = lasagne.layers.count_params(l_out)
print "  number of parameters: %d" % num_params

loss = -1.0 / batch_size * T.sum(mask_flat * T.log(predictions[T.arange(y.shape[0]), y]))
learning_rate = 0.5
updates = lasagne.updates.adadelta(loss, all_params, learning_rate)

iter_train = theano.function([idxs], [loss], updates=updates)

for _ in xrange(100):
    print iter_train(np.array([0, 2, 4], dtype='int32'))