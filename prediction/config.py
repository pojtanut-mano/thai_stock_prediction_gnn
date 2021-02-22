from datetime import datetime

# Readability config
verbose = 1

# Repeatability config
seed = 112

# Dataset config
feature_list = ['return']
index_col = ['Date']
market_directory = r'..\data\processed\price'
relation_directory = r'..\data\processed\relation'
adjacency_matrix_path_name = 'all_relation.npy'

path = r'checkpoint\18022021_13-23-05_best\TSGNN'

scale_type = 'normalize'
limiter = True

# Model config
mode = 'train'
model = 'TSGNN'
directory = model + '_' + datetime.now().strftime("%d%m%Y_%H-%M-%S")
lookback = 50
train_size = 250
valid_size = 50
test_size = 100
train_start = lookback

target_type = 'classification'  # classification or regression
num_sample_neighbors = 25

rank_weight = 0.3

## TSGNN config
checkpoint_dir = 'train_hist'
name = model + '.pt'

lstm_input_dims = len(feature_list)
lstm_hidden_dims = 64
lstm_layer = 1
lstm_dropout = 0.4

lr = 1e-3
loss_function = 'MSE'  # MSE, RankMSE or entropy
optimizer = 'RMSprop'
optimizer_weight_decay = 1e-6
gamma = 0.9
step_size = 5

clip_grad = 0

early_stopping_period = 10
early_stopping_threshold = 1e-6

overfitting_threshold = 0.03

# training config
epochs = 200
shuffle_batch = False
print_log = 10

# Export config
hist_dir = 'hist'
raw_output_dir = 'raw_output'
report_dir = 'report'
confusion_mat_dir = 'conf_mat'

fig_size = (10, 10)

