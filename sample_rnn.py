from __future__ import print_function

import os
import sys
import time
import importlib
if sys.version_info < (3,0):
    import cPickle as pickle
else:
    import pickle

from folk_rnn import Folk_RNN
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

if config.one_hot:
    config.embedding_size = None

folk_rnn = Folk_RNN(
    metadata['token2idx'],
    metadata['param_values'], 
    config.num_layers, 
    config.rnn_size,
    config.grad_clipping,
    config.dropout, 
    config.embedding_size, 
    rng_seed, 
    temperature
    )
folk_rnn.seed_tune(seed)
for i in xrange(ntunes):
    print(folk_rnn.compose_tune())
    
