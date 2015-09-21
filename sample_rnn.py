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
#     sys.exit("Usage: train_rnn.py <configuration_name> <metadata_path>")
#
# config_name = sys.argv[1]
# data_path = sys.argv[2]

config_name = 'config1'
metadata_path = 'metadata/config1-20150921-180629.pkl'

config = importlib.import_module('configurations.%s' % config_name)

with open(metadata_path) as f:
    metadata = pickle.load(f)

#TODO
vocab_size = 0

print 'Building the model'
l_inp = InputLayer((1, None))
l_emb = EmbeddingLayer(l_inp, input_size=vocab_size, output_size=config.embedding_size)

l_mask = InputLayer(shape=(1, None))

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