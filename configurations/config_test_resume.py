resume_path = 'metadata/config_test-input_test-20151001-130023.pkl'
max_epoch = 200  # increase max_epoch!

one_hot = True
embedding_size = 16
num_layers = 2
rnn_size = 32
dropout = 0.0

learning_rate = 0.003
learning_rate_decay_after = 800
learning_rate_decay = 0.97

batch_size = 2
grad_clipping = 5
validation_fraction = 0.2
validate_every = 100  # iterations

save_every = 100  # epochs