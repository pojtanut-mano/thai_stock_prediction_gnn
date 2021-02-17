from datetime import datetime

# Readability config
verbose = 1
directory = datetime.now().strftime("%d%m%Y_%H-%M-%S")

# Dataset config
feature_list = ['return']
index_col = ['Date']
market_directory = r'..\data\processed\price'
relation_directory = r'..\data\processed\relation'
adjacency_matrix_path_name = 'all_relation.npy'

scale_type = None

# Model config
lookback = 50
train_size = 250
valid_size = 50
test_size = 100
train_start = lookback

target_type = 'classification'  # classification or regression
num_sample_neighbors = 25

## TSGNN config
checkpoint_dir = 'checkpoint'
name = 'TSGNN'

lstm_input_dims = len(feature_list)
lstm_hidden_dims = 64
lstm_layer = 1
lstm_dropout = 0.4

lr = 0.001
loss_function = 'MSE'  # MSE, MSE_rank or entropy
optimizer = 'Adam'
optimizer_weight_decay = 1e-10
batch_size = 32
clip_grad = 0

early_stopping_period = 5
early_stopping_threshold = 1e-10


# training config
epochs = 300
shuffle_batch = True
print_log = 10
