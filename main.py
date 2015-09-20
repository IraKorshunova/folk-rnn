import theano
import lasagne
import sys
import importlib
import numpy as np
import theano.tensor as T
from lasagne.layers import *
from itertools import izip
from data_iter import DataIterator
from theano import config

#theano.config.optimizer='fast_compile'


#config_name = sys.argv[1]
#config = importlib.import_module("configurations.%s" % config_name)
config = importlib.import_module("configurations.config1")

with open('input_test.txt', 'r') as f:
    data = f.read()

tokens_set = set(data.split())
start_symbol, end_symbol = '<s>', '</s>'
tokens_set.update({start_symbol, end_symbol})

idx2token = list(tokens_set)
vocab_size = len(idx2token)
print 'vocabulary size:', vocab_size
token2idx = dict(izip(idx2token, xrange(vocab_size)))
tunes = data.split('\n\n')
del data

tunes = [[token2idx[c] for c in [start_symbol] + t.split() + [end_symbol]] for t in tunes]
tunes.sort(key=lambda x: len(x), reverse=True)

tune_lens = np.array([len(t) for t in tunes])
offsets = np.concatenate(([0], np.cumsum(tune_lens),))
max_len = max(tune_lens)

print tune_lens
print offsets

ntunes = len(tunes)
print 'n tunes:', ntunes
batches_per_epoch = ntunes / config.batch_size

tunes = np.array([i for tune in tunes for i in tune], dtype='float32')

print 'Load data to GPU'
data_shared = theano.shared(tunes)
tune_lens_shared = theano.shared(tune_lens)
offsets_shared = theano.shared(offsets)

batch_shared = theano.shared(np.zeros((config.batch_size, max_len), dtype='float32'))
mask_shared = theano.shared(np.zeros((config.batch_size, max_len - 1), dtype='float32'))

idxs = T.ivector('idxs')

itune_lens = T.cast(tune_lens_shared, 'int32')
ioffsets = T.cast(offsets_shared, 'int32')

for i in xrange(config.batch_size):
    j = idxs[i]
    batch_shared = T.set_subtensor(batch_shared[i, 0:itune_lens[j]],
                                   data_shared[ioffsets[j]:ioffsets[j + 1]])

    mask_shared = T.set_subtensor(mask_shared[i, 0:itune_lens[j] - 1], 1)
    mask_shared = T.set_subtensor(mask_shared[i, itune_lens[j] - 1:], 0)

max_seqlen = T.max(itune_lens[idxs])
x = T.cast(batch_shared[:, :max_seqlen - 1], 'int32')
y = T.cast(T.flatten(batch_shared[:, 1:max_seqlen]), 'int32')

mask = mask_shared[:, :max_seqlen-1]
mask_flat = T.flatten(mask)

l_inp = InputLayer((config.batch_size, None), input_var=x)
l_emb = EmbeddingLayer(l_inp, input_size=vocab_size, output_size=config.embedding_size)

l_mask = InputLayer(shape=(config.batch_size, None), input_var=mask)

recurrent_layers = []
for _ in xrange(config.num_layers):
    if not recurrent_layers:
        recurrent_layers.append(LSTMLayer(l_emb, num_units=config.rnn_size,
                                          peepholes=False, mask_input=l_mask,
                                          grad_clipping=config.grad_clipping))
    else:
        recurrent_layers.append(LSTMLayer(recurrent_layers[-1], num_units=config.rnn_size,
                                          peepholes=False, mask_input=l_mask,
                                          grad_clipping=config.grad_clipping))

l_reshp = ReshapeLayer(recurrent_layers[-1], (-1, config.rnn_size))
l_out = DenseLayer(l_reshp, num_units=vocab_size, nonlinearity=T.nnet.softmax)
predictions = lasagne.layers.get_output(l_out)

print "Build model"
all_layers = lasagne.layers.get_all_layers(l_out)
all_params = lasagne.layers.get_all_params(l_out)
num_params = lasagne.layers.count_params(l_out)
print 'number of parameters:', num_params

loss = -1.0 / config.batch_size * T.sum(mask_flat * T.log(predictions[T.arange(y.shape[0]), y]))

updates = lasagne.updates.adadelta(loss, all_params, config.learning_rate)

train = theano.function([idxs], [loss], updates=updates)

data_iter = DataIterator(tune_lens, config.batch_size, random_lens=False)

for niter, batch_idxs in enumerate(data_iter):
    if niter % batches_per_epoch == 0:
        print 'epoch'
    print batch_idxs
    print train(np.int32(batch_idxs))
    if niter == 1000:
        break