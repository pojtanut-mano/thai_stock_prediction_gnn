from datetime import datetime

# Readability config
verbose = 1

# Repeatability config
seed = 112

# Dataset config
full_trade_day = 1461
period = 2

feature_list = ['return']
index_col = ['Date']
reg_target_col = ['high_return']
market_directory = r'..\data\processed\price'
relation_directory = r'..\data\processed\relation'
adjacency_matrix_path_name = 'all_relation.npy'

path = r'checkpoint\18022021_13-23-05_best\TSGNN'

scale_type = 'normalize'
limiter = True

# Model config
mode = 'train'
model = 'LSTM'
directory = model + '_' + datetime.now().strftime("%d%m%Y_%H-%M-%S")
lookback = 50
train_size = 150
valid_size = 25
test_size = 25
train_start = lookback

target_type = 'classification'  # classification or regression

rank_weight = 0.3

# TSGNN config
checkpoint_dir = 'train_hist'
name = model + '.pt'

lstm_input_dims = len(feature_list)
lstm_layer = 1
lstm_dropout = 0.4

loss_function = 'MSE'  # MSE, RankMSE or entropy
gamma = 0.9
step_size = 5

clip_grad = 0

# TRS
relation_type = 'implicit'

# Hyper params search

c_param = {'num_sample_neighbors': [25, 30],
           'lstm_hidden_dims': [128],
           'optimizer': ['RMSprop', 'Adam'],
           'optimizer_weight_decay': [1e-4, 1e-6],
           'lr': [1e-3, 5e-4]}

c_grid = []
for a in c_param['num_sample_neighbors']:
    for b in c_param['lstm_hidden_dims']:
        for c in c_param['optimizer']:
            for d in c_param['optimizer_weight_decay']:
                for e in c_param['lr']:
                    c_grid.append([a, b, c, d, e])

early_stopping_period = 10
early_stopping_threshold = 1e-6

overfitting_threshold = 0.04

# training config
epochs = 200
shuffle_batch = False
print_log = 10

# Export config
hist_dir = 'hist'
raw_output_dir = 'raw_output'
report_dir = 'report'
confusion_mat_dir = 'conf_mat'
config_name = 'config.pkl'

fig_size = (10, 10)
