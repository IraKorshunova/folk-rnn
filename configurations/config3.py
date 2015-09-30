one_hot = True
embedding_size = 256  # is ignored if one_hot=True
num_layers = 3
rnn_size = 512
dropout = 0.5

learning_rate = 0.0025
learning_rate_decay_after = 15
learning_rate_decay = 0.97

batch_size = 50
max_epoch = 100
grad_clipping = 5
validation_fraction = 0.05
validate_every = 1000  # iterations

save_every = 10  # epochs

