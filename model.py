from ast import Pass
from collections import defaultdict
from pickle import FALSE
from stat import S_ISBLK
from tkinter import E
import dgl
from dgl.batch import batch
from dgl.dataloading import dataloader
from sklearn import neighbors
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv
from torch.utils import data
from torch import Tensor
from torch.autograd import Variable
from torch.nn import RNNBase
import math
import random

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    # 残差思想
    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

class GAT(nn.Module):
    def __init__(self, args, input_size, hidden_size, batch_size, poi_attention_coefficient, poi_neighbors):
        super(GAT, self).__init__()
        self.activation = torch.nn.ELU()
        self.poi_attention_coefficient = poi_attention_coefficient
        self.poi_neighbors = poi_neighbors  # 从 0 开始
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.device = args.device
        self.gconv = GATConv(input_size, hidden_size, num_heads=args.num_heads, feat_drop=args.feat_drop, attn_drop=args.attn_drop, residual=True, activation=self.activation)
        self.w = nn.Linear(input_size, hidden_size, bias=False)
    
    def get_poi_attention_coefficient(self, poi_i, poi_j):
        return self.poi_attention_coefficient[poi_i, poi_j]
    
    def forward(self, blocks, output_nodes, feat, poi_cat_feat):
        feat_gat, attn = self.gconv(blocks[0], feat, get_attention=True)
        feat_dist = torch.zeros(size=(feat_gat.shape[0], self.hidden_size)).to(self.device)
        
        for i, nid in enumerate(output_nodes):
            nid = int(nid)
            nid_neighbors = self.poi_neighbors[nid]
            current_feat = poi_cat_feat[nid].to(self.device)
            cnt = 0
            for j in nid_neighbors:
                beta = self.get_poi_attention_coefficient(nid, j)
                sub_feat = torch.mul(self.w(current_feat), beta)
                feat_dist[i] += sub_feat
                cnt += 1
                if cnt == 50:
                    break
        feat_dist = feat_dist.unsqueeze(1)
        feat_out = (feat_gat + feat_dist) / 2
        feat_out = F.relu(feat_out)
        return feat_out # (batch_size, num_heads, hidden_size) (128, 1, 200)

class SGLSP(nn.Module):
    def __init__(self, args, user_num, poi_num, time_num, cate_num, poi_adj=None, poi2cate=None, user_adj=None, user_poi_dict=None, poi_attention_coefficient=None, poi_neighbors=None):
        super(SGLSP, self).__init__()

        self.user_num = user_num        # user number
        self.poi_num = poi_num          # poi number
        self.time_num = time_num        # time number
        self.cate_num = cate_num        # category number

        self.poi_adj = poi_adj      # adj matrix
        self.poi2cate = poi2cate
        self.user_adj = user_adj
        self.user_poi_dict = user_poi_dict
        self.poi_attention_coefficient = poi_attention_coefficient
        self.poi_neighbors = poi_neighbors

        self.device = args.device       # device

        self.batch_size = args.batch_size
        self.hidden_size = args.hidden_size
        self.user_hidden_size = args.user_hidden_size
        self.poi_hidden_size = args.poi_hidden_size
        self.time_hidden_size = args.time_hidden_size
        self.cate_hidden_size = args.cate_hidden_size

        # self.all_hidden_size = args.user_hidden_size + args.poi_hidden_size + args.time_hidden_size + args.cate_hidden_size
        self.all_hidden_size = args.poi_hidden_size + args.time_hidden_size + args.cate_hidden_size
        
        # Embedding Layer
        self.user_embedding = nn.Embedding(self.user_num + 1, args.user_hidden_size, padding_idx=0)
        self.poi_embedding = nn.Embedding(self.poi_num + 1, args.poi_hidden_size, padding_idx=0)
        self.time_embedding = nn.Embedding(self.time_num + 1, args.time_hidden_size, padding_idx=0)
        self.cate_embedding = nn.Embedding(self.cate_num + 1, args.cate_hidden_size, padding_idx=0)
        self.embedding_dropout = nn.Dropout(args.dropout)

        ############################## MultiHeadAttention Layers ##############################
        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        for _ in range(args.num_layers):
            new_attn_layernorm = nn.LayerNorm(self.all_hidden_size, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)
            new_attn_layer = nn.MultiheadAttention(self.all_hidden_size, args.num_heads, args.dropout)
            # new_attn_layer = nn.Transformer(d_model=self.all_hidden_size, nhead=args.num_heads, dropout=args.dropout)
            self.attention_layers.append(new_attn_layer)
            new_fwd_layernorm = nn.LayerNorm(self.all_hidden_size, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)
            new_fwd_layer = PointWiseFeedForward(self.all_hidden_size, args.dropout)
            self.forward_layers.append(new_fwd_layer)
        self.last_layernorm = nn.LayerNorm(self.all_hidden_size, eps=1e-8)
        
        ############################## GNN Layers ##############################
        self.gnn_hidden_size = args.poi_hidden_size + args.cate_hidden_size
        self.gnn = GAT(args, self.gnn_hidden_size, self.gnn_hidden_size, self.batch_size, poi_attention_coefficient=self.poi_attention_coefficient, poi_neighbors=self.poi_neighbors)
        self.sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1) # sampler
        # self.sampler = dgl.dataloading.MultiLayerNeighborSampler([15]) # sampler
        self.geo_linear = nn.Linear(in_features=self.gnn_hidden_size, out_features=self.all_hidden_size)
        self.user_linear = nn.Linear(in_features=self.user_hidden_size, out_features=self.all_hidden_size)
        self.geo_layernorm = nn.LayerNorm(self.all_hidden_size, eps=1e-8)
        self.user_layernorm = nn.LayerNorm(self.all_hidden_size, eps=1e-8)

        ############################## RNN Layers ##############################
        self.rnn_type = args.rnn_type
        if self.rnn_type == "RNN":
            self.rnn = nn.RNN(input_size=self.all_hidden_size, hidden_size=self.all_hidden_size, num_layers=args.rnn_num_layers, 
                        dropout=args.dropout, bias=True, batch_first=True, bidirectional=False)
        elif self.rnn_type == "LSTM":
            self.rnn = nn.LSTM(input_size=self.all_hidden_size, hidden_size=self.all_hidden_size, num_layers=args.rnn_num_layers, 
                        dropout=args.dropout, bias=True, batch_first=True, bidirectional=False)
        elif self.rnn_type == "GRU":
            self.rnn = nn.GRU(input_size=self.all_hidden_size, hidden_size=self.all_hidden_size, num_layers=args.rnn_num_layers, 
                        dropout=args.dropout, bias=True, batch_first=True, bidirectional=False)
        self.output_layer = nn.Linear(self.all_hidden_size, self.poi_num)
        self.output_layer_cate = nn.Linear(self.all_hidden_size, self.cate_num)
    
    def build_geographical(self, pois):
        poi_nonzero = np.unique(pois)[1:] - 1
        poi_cat_feat = torch.cat([self.poi_embedding.weight[1:, :], self.cate_embedding.weight[[self.poi2cate[poi_id] for poi_id in range(1, self.poi_num + 1)], :]], dim=1)
        poi_cat_feat = poi_cat_feat.to("cpu")   # (batch_size, max_len, poi_hidden_size + cate_hidden_size)

        self.poi_adj.ndata["features"] = poi_cat_feat
        
        feat_dict = {}
        data_loader = dgl.dataloading.NodeDataLoader(
            self.poi_adj, poi_nonzero, self.sampler, batch_size=self.batch_size,
            shuffle=False, drop_last=False, num_workers=1)
        for input_nodes, output_nodes, blocks in data_loader:
            blocks = [b.to(self.device) for b in blocks]
            in_feat = blocks[0].srcdata["features"]

            out_feat = self.gnn(blocks, output_nodes, in_feat, poi_cat_feat)
            for i, nid in enumerate(set(output_nodes)):
                feat_dict[int(nid)] = out_feat[i].unsqueeze(0)
                
        (batch_size, max_len) = pois.shape
        feat = torch.zeros(size=(batch_size, max_len, self.gnn_hidden_size)).to(self.device)
        for i in range(batch_size):
            for j in range(max_len):
                nid = int(pois[i, j]) - 1
                if nid in feat_dict.keys():
                    feat[i, j, :] = feat_dict[nid]
        feat = self.embedding_dropout(feat)
        return feat
    
    def build_social(self, users, poi_emb, user_emb):
        (batch_size, max_len, _) = poi_emb.shape
        feat = torch.zeros(size=(batch_size, self.user_hidden_size)).to(self.device)
        for i in range(batch_size):
            uid = int(users[i])
            if uid == 0:
                continue
            if sum(self.user_adj[uid - 1]) == 0:
                feat[i, :] = user_emb[i, :].unsqueeze(0)
            else:
                neighbor_user = random.choice(np.nonzero(self.user_adj[uid - 1])[0])
                neighbor_poi = random.choice(self.user_poi_dict[neighbor_user])
                neighbor_user, neighbor_poi = torch.LongTensor([neighbor_user + 1]).to(self.device), torch.LongTensor([neighbor_poi + 1]).to(self.device)
                neighbor_user_emb = self.user_embedding(neighbor_user)
                neighbor_poi_emb = self.poi_embedding(neighbor_poi)
                current_user_emb = user_emb[i, :].unsqueeze(0)
                feat[i, :] = current_user_emb + neighbor_user_emb + neighbor_poi_emb
        feat = feat.unsqueeze(1).repeat(1, max_len, 1)
        feat = self.embedding_dropout(feat)
        return feat

    def forward(self, users, pois, times, cates):
        times = ((times / 3600) % self.time_num) + 1
        
        (batch_size, max_len) = pois.shape
        # (128, 50) (batch_size, max_len)
        user_emb = self.user_embedding(torch.LongTensor(users).to(self.device))
        # (128, 50, 50) (batch_size, max_len, hidden_dim)
        # user_emb = user_emb.unsqueeze(1).repeat(1, max_len, 1)
        poi_emb = self.poi_embedding(torch.LongTensor(pois).to(self.device))
        time_emb = self.time_embedding(torch.LongTensor(times).to(self.device))
        cate_emb = self.cate_embedding(torch.LongTensor(cates).to(self.device))

        # 长短期特征
        l_feat = torch.cat([poi_emb, time_emb, cate_emb], dim=2)
        s_feat = torch.cat([poi_emb, time_emb, cate_emb], dim=2)

        l_feat = self.embedding_dropout(l_feat)
        s_feat = self.embedding_dropout(s_feat)

        s_feat, _ = self.rnn(s_feat)

        tm = torch.BoolTensor(pois == 0).to(self.device)
        l_feat *= ~tm.unsqueeze(-1)
        am = ~torch.tril(torch.ones((max_len, max_len), dtype=torch.bool, device=self.device))
        for i in range(len(self.attention_layers)):
            l_feat = torch.transpose(l_feat, 0, 1)
            Q = self.attention_layernorms[i](l_feat)
            mha_feat, _ = self.attention_layers[i](Q, l_feat, l_feat, attn_mask=am)
            # mha_feat = self.attention_layers[i](Q, l_feat)   # Transformer
            l_feat = Q + mha_feat
            l_feat = torch.transpose(l_feat, 0, 1)
            l_feat = self.forward_layernorms[i](l_feat)
            l_feat = self.forward_layers[i](l_feat)
            l_feat *= ~tm.unsqueeze(-1)
        
        l_feat = self.last_layernorm(l_feat)
        s_feat = self.last_layernorm(s_feat)
        
        poi_geo_info = self.build_geographical(pois)    # (batch_size, max_len, hidden_size?)
        poi_geo_info = self.geo_linear(poi_geo_info)
        poi_geo_info = self.geo_layernorm(poi_geo_info)
        
        user_social_info = self.build_social(users, poi_emb, user_emb)
        user_social_info = self.user_linear(user_social_info)
        user_social_info = self.user_layernorm(user_social_info)        # (batch_size, max_len, user_hidden_size)
        
        feat = l_feat + s_feat + poi_geo_info + user_social_info
        
        feat_out = self.output_layer(feat)
        feat_out_cate = self.output_layer_cate(feat)

        return feat_out, feat_out_cate
