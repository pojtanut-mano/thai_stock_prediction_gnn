import torch
from torch import nn, optim
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
import random

EPSILON = 1e-10


class TSGNN(nn.Module):
    def __init__(self, config, neighbors, rel_num, hidden_dims, optimizer, weight_decay, lr, dropout_rate):
        super(TSGNN, self).__init__()
        self.config = config
        self.neighbors = neighbors
        self.rel_num = rel_num
        self.checkpoint_directory = config.checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_directory, config.directory, config.name)

        self.hidden_dims = hidden_dims
        self.relation_weight = None

        # reproducibility
        self.init_seed()

        self.lstm = nn.LSTM(config.lstm_input_dims, hidden_dims,
                            config.lstm_layer)
        self.dropout = nn.Dropout(p=dropout_rate)

        # Attention layer functions
        self.fc_att = nn.Linear(in_features=2 * hidden_dims + neighbors.shape[0],
                                out_features=1)
        self.att_softmax = nn.Softmax(dim=2)

        # Relation layer functions
        self.fc_rel_att = nn.Linear(in_features=2 * hidden_dims + neighbors.shape[0],
                                    out_features=1)
        self.rel_att_softmax = nn.Softmax(dim=0)

        # Fully-connected layer
        if config.target_type == 'classification':
            self.fc1 = nn.Linear(in_features=hidden_dims,
                                 out_features=3)
        else:
            self.fc1 = nn.Linear(in_features=hidden_dims,
                                 out_features=1)
        self.softmax = nn.Softmax(dim=1)

        # Utils
        if optimizer == 'Adam':
            self.optimizer = optim.Adam(params=self.parameters(), lr=lr,
                                        weight_decay=weight_decay)
        else:
            self.optimizer = optim.RMSprop(params=self.parameters(), lr=lr,
                                           weight_decay=weight_decay)
        if config.target_type == 'classification':
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = nn.MSELoss()

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

        # Device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Current device: {}\n'.format(self.device))
        self.to(self.device)

        # Weight init
        self.apply(self.weight_init)

        # Clip gradient
        if config.clip_grad > 0:
            clip_grad_norm_(self.parameters(), config.clip_grad)

        # lr scheduler
        self.scheduler = StepLR(self.optimizer, step_size=self.config.step_size,
                                gamma=self.config.gamma)

    def forward(self, stock_hist):
        _, (state_embedding, _) = self.lstm(stock_hist)
        state_embedding = self.dropout(state_embedding)
        state_embedding = torch.squeeze(state_embedding, dim=0)
        # print(torch.isnan(state_embedding).any())

        # Padding 0 as a placeholder for nodes that don't have relation
        state_embedding = torch.cat((torch.zeros(1, self.hidden_dims).to(self.device),
                                     state_embedding), 0)

        updated_state_embedding, self.relation_weight = self.graph_attention_layer(state_embedding)
        final_state_embedding = updated_state_embedding + state_embedding[1:]

        # Prediction layer
        preds = self.fc1(final_state_embedding)
        if self.config.target_type == 'classification':
            preds = self.softmax(preds)

        return preds

    def graph_attention_layer(self, state_embedding):
        # Look up each neighbor
        embedding_lookup = nn.Embedding.from_pretrained(state_embedding)
        neighbors_embedding = embedding_lookup(self.neighbors.to(self.device))

        #########################
        # State attention layer #
        #########################

        # Create relation embedding
        att_rel_emb = self.create_relation_embedding(neighbors_embedding.shape,
                                                     layer='attention')

        # Create holder for company i (itself, not neighbors)
        self_emb = torch.unsqueeze(torch.unsqueeze(state_embedding[1:], 1), 0)
        self_emb = self_emb.repeat(neighbors_embedding.shape[0], 1, neighbors_embedding.shape[2], 1)

        # Concat (neighbors embedding, self embedding, relation embedding)
        attention = torch.cat((neighbors_embedding, self_emb, att_rel_emb), -1)
        attention_score = self.leaky_relu(self.fc_att(attention))
        attention_weight = self.att_softmax(attention_score)
        # print(attention_weight.shape, neighbors_embedding.shape)

        # Aggregate score
        relation_count = torch.from_numpy(np.expand_dims(self.rel_num + EPSILON, -1)).type(torch.float32).to(
            self.device)
        relation_representation = torch.sum(attention_weight * neighbors_embedding, dim=2) / relation_count

        ############################
        # Relation attention layer #
        ############################

        # create relation embedding for relation attention layer
        rel_att_rel_emb = self.create_relation_embedding(dims=neighbors_embedding.shape,
                                                         layer='relation')

        # Create node embedding for relation attention
        rel_att_self_emb = torch.unsqueeze(state_embedding[1:], 0)
        rel_att_self_emb = rel_att_self_emb.repeat(neighbors_embedding.shape[0], 1, 1)

        relation_attention = torch.cat((relation_representation, rel_att_rel_emb, rel_att_self_emb), -1)

        # Concat (relation representation, relation embedding)
        relation_attention_score = self.leaky_relu(self.fc_rel_att(relation_attention))
        relation_attention_weight = self.rel_att_softmax(relation_attention_score)

        updated_state_embedding = torch.mean(relation_attention_weight * relation_representation,
                                             dim=0)
        return updated_state_embedding, torch.squeeze(relation_attention_weight, dim=-1)

    def create_relation_embedding(self, dims, layer):
        num_relations = dims[0]
        one_hot = []
        for i in range(num_relations):
            row_one_hot = np.zeros(num_relations)
            row_one_hot[i] = 1
            one_hot.append(row_one_hot)
        one_hot = np.stack(one_hot, axis=0)
        emb_ = []
        if layer == 'attention':
            for i in range(one_hot.shape[0]):
                exp = np.tile(np.expand_dims(np.expand_dims(one_hot[i], 0), 0), [dims[1], dims[2], 1])
                emb_.append(exp)
            emb_ = np.stack(emb_, axis=0)
            rel_emb = torch.from_numpy(emb_).type(torch.float32).to(self.device)
        elif layer == 'relation':
            for i in range(one_hot.shape[0]):
                exp = np.tile(np.expand_dims(one_hot[i], 0), [dims[1], 1])
                emb_.append(exp)
            emb_ = np.stack(emb_, axis=0)
            rel_emb = torch.from_numpy(emb_).type(torch.float32).to(self.device)
        return rel_emb

    def weight_init(self, m):
        """Input: m as a class of pytorch nn Module
           Initialize """
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def get_relation_weight(self):
        return self.relation_weight.detach().cpu().numpy()

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file))

    def load(self, path):
        print('... loading from path ...')
        self.load_state_dict(torch.load(path))

    def init_seed(self):
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        random.seed(self.config.seed)

    def save_model(self):
        print('... saving state ...')
        return self.state_dict()

    def load_state(self, state):
        print('Loading best params...')
        self.load_state_dict(state)
