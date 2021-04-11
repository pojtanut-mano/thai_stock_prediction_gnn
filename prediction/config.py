from datetime import datetime

# Readability config
verbose = 1

# Repeatability config
seed = 112

# Dataset config
full_trade_day = 1461

feature_list = ['Open', 'High', 'Low', 'Close', 'Volume', 'return_feat']
index_col = ['Date']
target_col = ['return_target']
market_directory = r'..\data\processed\price'
relation_directory = r'..\data\processed\relation'
adjacency_matrix_path_name = 'all_stock_all_relation.npy'
ticker_file = 'all_tickers.pkl'

target_look_forward = 1

# Model config
model = 'MLP'
directory = model + '_' + datetime.now().strftime("%d%m%Y_%H-%M-%S")

lookback = 50
train_size = 150
valid_size = 25
test_size = 50
train_start = lookback

# TSGNN config
checkpoint_dir = 'train_hist'
name = model + '.pt'

lstm_input_dims = len(feature_list)
lstm_layer = 1

gamma = 0.9
step_size = 5

clip_grad = 0

# TRS
relation_type = 'implicit'

# Hyper params search
c_param = {'num_sample_neighbors': [25],
           'lstm_hidden_dims': [128],
           'optimizer': ['Adam', 'RMSprop'],
           'optimizer_weight_decay': [1e-6],
           'lr': [1e-3, 5e-4],
           'dropout_rate': [0.1, 0.3]}

c_grid = []
for a in c_param['num_sample_neighbors']:
    for b in c_param['lstm_hidden_dims']:
        for c in c_param['optimizer']:
            for d in c_param['optimizer_weight_decay']:
                for e in c_param['lr']:
                    for f in c_param['dropout_rate']:
                        c_grid.append([a, b, c, d, e, f])

early_stopping_period = 5
early_stopping_threshold = 1e-6

overfitting_threshold = 0.04

# training config
epochs = 1
shuffle_batch = True
print_log = 10

# Export config
hist_dir = 'hist'
raw_output_dir = 'raw_output'
report_dir = 'report'
confusion_mat_dir = 'conf_mat'
config_name = 'config.pkl'

fig_size = (10, 10)

start_period = 0
end_period = 24

callback_period = 30

scale_type = 'normalize'
