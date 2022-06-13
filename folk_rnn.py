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

class Folk_RNN:
    """
    Folk music style modelling using LSTMs
    """
    
    debug = False
    
    def __init__(self, 
                 token2idx, 
                 param_values,
                 num_layers,
                 rnn_size, 
                 grad_clipping,
                 dropout=0, 
                 embedding_size=None, 
                 random_number_generator_seed=42, 
                 temperature=1.0
                 ):
        theano.config.floatX = 'float64'
        vocab_size = len(token2idx)
        one_hot = embedding_size is None
        
        self.token2idx = token2idx
        self.idx2token = dict((v, k) for k, v in self.token2idx.iteritems())
        self.vocab_idxs = np.arange(vocab_size)
        self.start_idx, self.end_idx = self.token2idx['<s>'], self.token2idx['</s>']
        self.rng = np.random.RandomState(random_number_generator_seed)
        self.seed_tune()
       
        if self.debug:
            print('Building the model')
        
        x = T.imatrix('x')
        
        l_inp = InputLayer((1, None), input_var=x)
        
        W_emb = np.eye(vocab_size, dtype='float32') if one_hot else lasagne.init.Orthogonal()
        emb_output_size = vocab_size if one_hot else embedding_size
        l_emb = EmbeddingLayer(l_inp, input_size=vocab_size, output_size=emb_output_size, W=W_emb)
        
        main_layers = []
        for _ in xrange(num_layers):
            if not main_layers:
                main_layers.append(LSTMLayer(l_emb, num_units=rnn_size,
                                             peepholes=False,
                                             grad_clipping=grad_clipping))
            else:
                main_layers.append(LSTMLayer(main_layers[-1], num_units=rnn_size,
                                             peepholes=False,
                                             grad_clipping=grad_clipping))
            if dropout > 0:
                main_layers.append(DropoutLayer(main_layers[-1], p=dropout))
        
        l_reshp = ReshapeLayer(main_layers[-1], (-1, rnn_size))
        l_out = DenseLayer(l_reshp, num_units=vocab_size, nonlinearity=lasagne.nonlinearities.identity)
        predictions = T.nnet.softmax(lasagne.layers.get_output(l_out, deterministic=True)[-1, :] / temperature)[0]
        lasagne.layers.set_all_param_values(l_out, param_values)
        
        if self.debug:
            all_layes = lasagne.layers.get_all_layers(l_out)
            for layer in all_layes:
                print(layer.__class__.__name__)
                for p in layer.get_params():
                    print(p.get_value().shape)
            print('number of parameters: ', lasagne.layers.count_params(l_out))
        
        self.predict = theano.function([x], predictions)
    
    def seed_tune(self, seed_tune_abc=None):
        """
        Sets the seed of the tune
        """
        self.tune = [self.start_idx]
        if seed_tune_abc is not None:
            self.tune += [self.token2idx[x] for x in seed_tune_abc.split(' ')]
    
    def compose_tune(self):
        """
        Composes tune and returns it as abc notation
        """
        tune = list(self.tune)
        while tune[-1] != self.end_idx:
            next_itoken = self.rng.choice(self.vocab_idxs, p=self.predict(np.array([tune], dtype='int32')))
            tune.append(next_itoken)
        return '{}\n{}\n{}\n'.format(
            self.idx2token[tune[1]],
            self.idx2token[tune[2]],
            ' '.join(self.idx2token[x] for x in tune[3:-1]),
            )
