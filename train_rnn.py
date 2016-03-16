import sys
import os
import time
import string
import logger
import theano
import importlib
import numpy as np
import lasagne as nn
import cPickle as pickle
from itertools import izip
import theano.tensor as T
from lasagne.layers import *
from data_iter import DataIterator

print theano.config.floatX
theano.config.warn_float64 = 'raise'

if len(sys.argv) < 3:
    sys.exit("Usage: train_rnn.py <configuration_name> <train data filename>")

config_name = sys.argv[1]
data_path = sys.argv[2]

config = importlib.import_module('configurations.%s' % config_name)
experiment_id = '%s-%s-%s' % (
    config_name.split('.')[-1], os.path.basename(data_path.split('.')[0]),
    time.strftime("%Y%m%d-%H%M%S", time.localtime()))
print experiment_id

# metadata
if not os.path.isdir('metadata'):
        os.makedirs('metadata')
metadata_target_path = 'metadata/%s.pkl' % experiment_id

# logs
if not os.path.isdir('logs'):
        os.makedirs('logs')
sys.stdout = logger.Logger('logs/%s.log' % experiment_id)
sys.stderr = sys.stdout

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
ntunes = len(tunes)

tune_lens = np.array([len(t) for t in tunes])
max_len = max(tune_lens)

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
print 'min, max length', min(tune_lens), max(tune_lens)

print 'Building the model'
x = T.fmatrix('x')
mask = T.matrix('mask')

l_inp = InputLayer((config.batch_size, None), input_var=T.cast(x, 'int32'))

W_emb = np.eye(vocab_size, dtype='float32') if config.one_hot else nn.init.Orthogonal()
emb_output_size = vocab_size if config.one_hot else config.embedding_size
l_emb = EmbeddingLayer(l_inp, input_size=vocab_size, output_size=emb_output_size, W=W_emb)

l_mask = InputLayer(shape=(config.batch_size, None), input_var=mask)

main_layers = []
for _ in xrange(config.num_layers):
    if not main_layers:
        input_gate = Gate(W_in=nn.init.GlorotUniform(), W_hid=nn.init.Orthogonal())
        forget_gate = Gate(W_in=nn.init.GlorotUniform(), W_hid=nn.init.Orthogonal(), b=nn.init.Constant(5.0))
        output_gate = Gate(W_in=nn.init.GlorotUniform(), W_hid=nn.init.Orthogonal())
        cell = Gate(W_in=nn.init.GlorotUniform(), W_hid=nn.init.Orthogonal(), W_cell=None,
                    nonlinearity=nn.nonlinearities.tanh)

        main_layers.append(LSTMLayer(l_emb, num_units=config.rnn_size,
                                     hid_init=nn.init.Orthogonal(),
                                     ingate=input_gate, forgetgate=forget_gate,
                                     cell=cell, outgate=output_gate,
                                     peepholes=False,
                                     mask_input=l_mask,
                                     precompute_input=False,
                                     grad_clipping=config.grad_clipping))
    else:
        input_gate = Gate(W_in=nn.init.GlorotUniform(), W_hid=nn.init.Orthogonal())
        forget_gate = Gate(W_in=nn.init.GlorotUniform(), W_hid=nn.init.Orthogonal(), b=nn.init.Constant(5.0))
        output_gate = Gate(W_in=nn.init.GlorotUniform(), W_hid=nn.init.Orthogonal())
        cell = Gate(W_in=nn.init.GlorotUniform(), W_hid=nn.init.Orthogonal(), W_cell=None,
                    nonlinearity=nn.nonlinearities.tanh)

        main_layers.append(LSTMLayer(main_layers[-1], num_units=config.rnn_size,
                                     hid_init=nn.init.Orthogonal(),
                                     ingate=input_gate, forgetgate=forget_gate,
                                     cell=cell, outgate=output_gate,
                                     peepholes=False,
                                     mask_input=l_mask,
                                     precompute_input=False,
                                     grad_clipping=config.grad_clipping))
    if config.dropout > 0:
        main_layers.append(DropoutLayer(main_layers[-1], p=config.dropout))

l_reshp = ReshapeLayer(main_layers[-1], (-1, config.rnn_size))
l_out = DenseLayer(l_reshp, num_units=vocab_size, W=nn.init.Orthogonal(), nonlinearity=T.nnet.softmax)
predictions = nn.layers.get_output(l_out)
predictions_det = nn.layers.get_output(l_out, deterministic=True)

all_params = nn.layers.get_all_params(l_out)
if config.one_hot:
    all_params = all_params[1:]
all_layers = nn.layers.get_all_layers(l_out)
num_params = nn.layers.count_params(l_out)
print '  number of parameters: %d' % num_params
print string.ljust('  layer output shapes:', 36),
print string.ljust('#params:', 10),
print 'output shape:'
for layer in all_layers:
    name = string.ljust(layer.__class__.__name__, 32)
    num_param = sum([np.prod(p.get_value().shape) for p in layer.get_params()])
    num_param = string.ljust(num_param.__str__(), 10)
    print '    %s %s %s' % (name, num_param, layer.output_shape)


y = T.cast(T.flatten(x[:, 1:]), 'int32')
# training loss
p1 = T.reshape(T.log(predictions[T.arange(y.shape[0]), y]), mask.shape)
loss = -1. * T.mean(T.sum(mask * p1, axis=1), axis=0)

# validation loss (with disabled dropout)
p1_det = T.reshape(T.log(predictions_det[T.arange(y.shape[0]), y]), mask.shape)
loss_det = -1. * T.mean(T.sum(mask * p1_det, axis=1), axis=0)


learning_rate = theano.shared(np.float32(config.learning_rate))
grads = theano.grad(loss, all_params)
updates = nn.updates.rmsprop(grads, all_params, config.learning_rate)

train = theano.function([x, mask], loss, updates=updates)
validate = theano.function([x, mask], loss_det)


def create_batch(idxs):
    max_seq_len = max([len(tunes[i]) for i in idxs])
    x = np.zeros((config.batch_size, max_seq_len), dtype='float32')
    mask = np.zeros((config.batch_size, max_seq_len - 1), dtype='float32')
    for i, j in enumerate(idxs):
        x[i, :tune_lens[j]] = tunes[j]
        mask[i, : tune_lens[j] - 1] = 1
    return x, mask


train_data_iterator = DataIterator(tune_lens[train_idxs], train_idxs, config.batch_size, random_lens=False)
valid_data_iterator = DataIterator(tune_lens[valid_idxs], valid_idxs, config.batch_size, random_lens=False)

print 'Train model'
train_batches_per_epoch = ntrain_tunes / config.batch_size
max_niter = config.max_epoch * train_batches_per_epoch
losses_train = []

nvalid_batches = nvalid_tunes / config.batch_size
losses_eval_valid = []
niter = 1
start_epoch = 0
prev_time = time.clock()

if hasattr(config, 'resume_path'):
    print 'Load metadata for resuming'
    with open(config.resume_path) as f:
        resume_metadata = pickle.load(f)

    nn.layers.set_all_param_values(l_out, resume_metadata['param_values'])
    start_epoch = resume_metadata['epoch_since_start'] + 1
    niter = resume_metadata['iters_since_start']
    learning_rate.set_value(resume_metadata['learning_rate'])
    print 'setting learning rate to %.7f' % resume_metadata['learning_rate']

for epoch in xrange(start_epoch, config.max_epoch):
    for train_batch_idxs in train_data_iterator:
        x_batch, mask_batch = create_batch(train_batch_idxs)
        train_loss = train(x_batch, mask_batch)
        current_time = time.clock()

        print '%d/%d (epoch %.3f) train_loss=%6.8f time/batch=%.2fs' % (
            niter, max_niter, niter / float(train_batches_per_epoch), train_loss, current_time - prev_time)

        prev_time = current_time
        losses_train.append(train_loss)
        niter += 1

        if niter % config.validate_every == 0:
            print 'Validating'
            avg_valid_loss = 0
            for valid_batch_idx in valid_data_iterator:
                x_batch, mask_batch = create_batch(valid_batch_idx)
                avg_valid_loss += validate(x_batch, mask_batch)
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
                'param_values': nn.layers.get_all_param_values(l_out),
            }, f, pickle.HIGHEST_PROTOCOL)

        print "  saved to %s" % metadata_target_path
