from __future__ import print_function

import theano
import lasagne
import os
import sys
import time
import importlib
if sys.version_info < (3,0):
    import cPickle as pickle
else:
    import pickle
import numpy as np
import theano.tensor as T
from lasagne.layers import *

theano.config.floatX = 'float64'

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('metadata_path')
parser.add_argument('--rng_seed', type=int, default=42)
parser.add_argument('--temperature', type=float, default=1.0)
parser.add_argument('--ntunes', type=int, default=1)
parser.add_argument('--seed')
parser.add_argument('--terminal', action="store_true")

args = parser.parse_args()

metadata_path = args.metadata_path
rng_seed = args.rng_seed
temperature = args.temperature
ntunes = args.ntunes
seed = args.seed

with open(metadata_path) as f:
    metadata = pickle.load(f)

config = importlib.import_module('configurations.%s' % metadata['configuration'])

# samples dir
if not os.path.isdir('samples'):
        os.makedirs('samples')
target_path = "samples/%s-s%d-%.2f-%s.txt" % (
    metadata['experiment_id'], rng_seed, temperature, time.strftime("%Y%m%d-%H%M%S", time.localtime()))

token2idx = metadata['token2idx']
idx2token = dict((v, k) for k, v in token2idx.iteritems())
vocab_size = len(token2idx)

print('Building the model')
x = T.imatrix('x')

l_inp = InputLayer((1, None), input_var=x)

W_emb = np.eye(vocab_size, dtype='float32') if config.one_hot else lasagne.init.Orthogonal()
emb_output_size = vocab_size if config.one_hot else config.embedding_size
l_emb = EmbeddingLayer(l_inp, input_size=vocab_size, output_size=emb_output_size, W=W_emb)

main_layers = []
for _ in xrange(config.num_layers):
    if not main_layers:
        main_layers.append(LSTMLayer(l_emb, num_units=config.rnn_size,
                                     peepholes=False,
                                     grad_clipping=config.grad_clipping))
    else:
        main_layers.append(LSTMLayer(main_layers[-1], num_units=config.rnn_size,
                                     peepholes=False,
                                     grad_clipping=config.grad_clipping))
    if config.dropout > 0:
        main_layers.append(DropoutLayer(main_layers[-1], p=config.dropout))

l_reshp = ReshapeLayer(main_layers[-1], (-1, config.rnn_size))
l_out = DenseLayer(l_reshp, num_units=vocab_size, nonlinearity=lasagne.nonlinearities.identity)
predictions = T.nnet.softmax(lasagne.layers.get_output(l_out, deterministic=True)[-1, :] / temperature)[0]

all_params = lasagne.layers.get_all_params(l_out)
lasagne.layers.set_all_param_values(l_out, metadata['param_values'])

all_layes = lasagne.layers.get_all_layers(l_out)
for layer in all_layes:
    print(layer.__class__.__name__)
    for p in layer.get_params():
        print(p.get_value().shape)
print('number of parameters: ', lasagne.layers.count_params(l_out))

predict = theano.function([x], predictions)

start_idx, end_idx = token2idx['<s>'], token2idx['</s>']

rng = np.random.RandomState(rng_seed)
vocab_idxs = np.arange(vocab_size)

# Converting the seed passed as an argument into a list of idx
seed_sequence = [start_idx]
if seed is not None:
    for token in seed.split(' '):
        seed_sequence.append(token2idx[token])

for i in xrange(ntunes):
    sequence = seed_sequence[:]
    while sequence[-1] != end_idx:
        next_itoken = rng.choice(vocab_idxs, p=predict(np.array([sequence], dtype='int32')))
        sequence.append(next_itoken)

    abc_tune = [idx2token[j] for j in sequence[1:-1]]
    if not args.terminal:
        f = open(target_path, 'a+')
	f.write('X:' + repr(i) + '\n')
        f.write(abc_tune[0] + '\n')
        f.write(abc_tune[1] + '\n')
        f.write(' '.join(abc_tune[2:]) + '\n\n')
        f.close()
    else:
	print('X:' + repr(i))
        print(abc_tune[0])
        print(abc_tune[1])
        print(' '.join(abc_tune[2:]) + '\n')


if not args.terminal:
    print('Saved to '+target_path)
