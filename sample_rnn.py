import theano
import lasagne
import sys
import time
import os
import importlib
import cPickle as pickle
import numpy as np
import theano.tensor as T
from lasagne.layers import *
from itertools import izip
from data_iter import DataIterator

metadata_path = 'metadata/config1-20150922-192341.pkl'
method = 'argmax'
temperature = ''

# if not (2 <= len(sys.argv) <= 4):
#     sys.exit("Usage: sample_rnn.py <metadata_path> [sampling temperature]")
#
# metadata_path = sys.argv[1]
#
# if len(sys.argv) == 3:
#     method = 'sampling'
#     temperature = sys.argv[2]
# else:
#     method = 'argmax'
#     temperature = ''

with open(metadata_path) as f:
    metadata = pickle.load(f)

config = importlib.import_module('configurations.%s' % metadata['configuration'])
print metadata['losses_train']

target_path = "samples/%s_%s%s.txt" % (os.path.basename(metadata_path).split('.')[0], method, str(temperature))

token2idx = metadata['token2idx']
idx2token = dict((v, k) for k, v in token2idx.iteritems())
vocab_size = len(token2idx)

print 'Building the model'
x = T.imatrix('x')
l_inp = InputLayer((1, None), input_var=x)
l_emb = EmbeddingLayer(l_inp, input_size=vocab_size, output_size=config.embedding_size)

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
l_out = DenseLayer(l_reshp, num_units=vocab_size, nonlinearity=T.nnet.softmax)
predictions = lasagne.layers.get_output(l_out)[-1, :]

all_params = lasagne.layers.get_all_params(l_out)
num_params = lasagne.layers.count_params(l_out)
print 'number of parameters:', num_params

lasagne.layers.set_all_param_values(l_out, metadata['param_values'])

predict = theano.function([x], predictions)

start_idx, end_idx = token2idx['<s>'], token2idx['</s>']

sequence = [start_idx]
while sequence[-1] != end_idx:
    next_itoken = np.argmax(predict(np.array([sequence], dtype='int32')))
    sequence.append(next_itoken)
    print idx2token[next_itoken]