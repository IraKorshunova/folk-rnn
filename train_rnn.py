import theano
import lasagne
import sys
import time
import importlib
import cPickle as pickle
import numpy as np
import theano.tensor as T
from lasagne.layers import *
from itertools import izip
from data_iter import DataIterator

# if len(sys.argv) < 3:
#     sys.exit("Usage: train_rnn.py <configuration_name> <train data filename>")
#
# config_name = sys.argv[1]
# data_path = sys.argv[2]

config_name = 'config1'
data_path = 'data/input_test.txt'

config = importlib.import_module('configurations.%s' % config_name)

experiment_id = '%s-%s' % (config_name.split('.')[-1], time.strftime("%Y%m%d-%H%M%S", time.localtime()))
metadata_target_path = 'metadata/%s.pkl' % experiment_id

with open(data_path, 'r') as f:
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

ntunes = len(tunes)

nvalid_tunes = ntunes * config.validation_fraction
nvalid_tunes = config.batch_size * max(1, np.rint(
    nvalid_tunes / float(config.batch_size)))  # round to the multiple of batch_size

rng = np.random.RandomState(42)
valid_idxs = rng.choice(np.arange(ntunes), nvalid_tunes, replace=False)

ntrain_tunes = ntunes - nvalid_tunes
train_idxs = np.delete(np.arange(ntunes), valid_idxs)

print 'n tunes:', ntunes
print 'n train tunes:', ntrain_tunes
print 'n validation tunes:', nvalid_tunes

tunes = np.array([i for tune in tunes for i in tune], dtype='float32')

print 'Load data to', theano.config.device
data_shared = theano.shared(tunes)
tune_lens_shared = theano.shared(tune_lens)
offsets_shared = theano.shared(offsets)
batch_shared = theano.shared(np.zeros((config.batch_size, max_len), dtype='float32'))
mask_shared = theano.shared(np.zeros((config.batch_size, max_len - 1), dtype='float32'))

print 'Building the model'
idxs = T.ivector('idxs')

itune_lens = T.cast(tune_lens_shared, 'int32')
ioffsets = T.cast(offsets_shared, 'int32')

for i in xrange(config.batch_size):
    j = idxs[i]
    batch_shared = T.set_subtensor(batch_shared[i, :itune_lens[j]],
                                   data_shared[ioffsets[j]:ioffsets[j + 1]])

    mask_shared = T.set_subtensor(mask_shared[i, :itune_lens[j] - 1], 1)
    mask_shared = T.set_subtensor(mask_shared[i, itune_lens[j] - 1:], 0)

max_seqlen = T.max(itune_lens[idxs])
x = T.cast(batch_shared[:, :max_seqlen - 1], 'int32')
y = T.cast(T.flatten(batch_shared[:, 1:max_seqlen]), 'int32')
mask = mask_shared[:, :max_seqlen - 1]
mask_flat = T.flatten(mask)

l_inp = InputLayer((config.batch_size, None), input_var=x)
l_emb = EmbeddingLayer(l_inp, input_size=vocab_size, output_size=config.embedding_size)

l_mask = InputLayer(shape=(config.batch_size, None), input_var=mask)

main_layers = []
for _ in xrange(config.num_layers):
    if not main_layers:
        main_layers.append(LSTMLayer(l_emb, num_units=config.rnn_size,
                                     peepholes=False, mask_input=l_mask,
                                     grad_clipping=config.grad_clipping))
    else:
        main_layers.append(LSTMLayer(main_layers[-1], num_units=config.rnn_size,
                                     peepholes=False, mask_input=l_mask,
                                     grad_clipping=config.grad_clipping))
    if config.dropout > 0:
        main_layers.append(DropoutLayer(main_layers[-1], p=config.dropout))

l_reshp = ReshapeLayer(main_layers[-1], (-1, config.rnn_size))
l_out = DenseLayer(l_reshp, num_units=vocab_size, nonlinearity=T.nnet.softmax)
predictions = lasagne.layers.get_output(l_out)

all_params = lasagne.layers.get_all_params(l_out)
num_params = lasagne.layers.count_params(l_out)
print 'number of parameters:', num_params

# do something with predictions to calculate loss
p1 = T.reshape(T.log(predictions[T.arange(y.shape[0]), y]), (config.batch_size, max_seqlen-1))
p2 = T.sum(mask*p1, axis=1)/itune_lens[idxs]
loss = -1.0/config.batch_size * T.sum(p2)

learning_rate = theano.shared(np.float32(config.learning_rate))

updates = lasagne.updates.rmsprop(loss, all_params, config.learning_rate)

train = theano.function([idxs], loss, updates=updates)
validate = theano.function([idxs], loss)

train_data_iterator = DataIterator(tune_lens[train_idxs], train_idxs, config.batch_size, random_lens=False)
valid_data_iterator = DataIterator(tune_lens[valid_idxs], valid_idxs, config.batch_size, random_lens=False)

print 'Train model'
train_batches_per_epoch = ntrain_tunes / config.batch_size
max_niter = config.max_epoch * train_batches_per_epoch
losses_train = []

nvalid_batches = nvalid_tunes / config.batch_size
losses_eval_valid = []

niter = 1
prev_time = time.clock()

for epoch in xrange(config.max_epoch):
    for train_batch_idxs in train_data_iterator:
        train_loss = train(np.int32(train_batch_idxs))
        iter_time = time.clock() - prev_time

        grad_param_norm = 0 # TODO
        print '%d/%d (epoch %.3f) train_loss=%6.8f  grad/param_norm=%6.4e time/batch=%.2fs' % (
            niter, max_niter, niter / float(train_batches_per_epoch), train_loss, grad_param_norm, iter_time)

        prev_time = iter_time
        losses_train.append(train_loss)
        niter += 1

        if niter % config.validate_every == 0:
            print 'Validating'
            avg_valid_loss = 0
            for valid_batch_idx in valid_data_iterator:
                avg_valid_loss += validate(valid_batch_idx)
            avg_valid_loss /= nvalid_batches
            losses_eval_valid.append(avg_valid_loss)
            print "    loss:\t%.6f" % avg_valid_loss
            print

    if epoch > config.learning_rate_decay_after:
        new_learning_rate = np.float32(learning_rate.get_value() * config.learning_rate_decay)
        learning_rate.set_value(new_learning_rate)
        print 'setting learning rate to %.7f' % new_learning_rate

    if (epoch + 1) % config.save_every == 0:
        with open(metadata_target_path, 'w') as f:
            pickle.dump({
                'configuration': config_name,
                'experiment_id': experiment_id,
                'epoch_since_start': epoch,
                'iters_since_start': niter,
                'losses_train': losses_train,
                'losses_eval_valid': losses_eval_valid,
                'learning_rate': learning_rate.get_value(),
                'token2idx': token2idx,
                # 'losses_eval_train': losses_eval_train,
                'param_values': lasagne.layers.get_all_param_values(l_out),
            }, f, pickle.HIGHEST_PROTOCOL)

        print "  saved to %s" % metadata_target_path
