# Readability config
verbose = 1

# Dataset config
feature_list = ['return']
index_col = ['Date']
market_directory = r'..\data\processed\price'
relation_directory = r'..\data\processed\relation'

scale_type = None

# Model config
lookback = 50
train_size = 250
valid_size = 50
test_size = 100
train_start = lookback

## TSGNN config
checkpoint_dir = 'checkpoint'
name = 'TSGNN'

lstm_input_dims = len(feature_list)
lstm_hidden_dims = 64
lstm_layer = 1

lr = 1e-3

# training config
epochs = 5
