#!/usr/bin/env python
# Author  : KerryChen
# File    : model.py
# Time    : 2025/7/17 11:30

import torch.nn.functional as F
from dgllife.model import GCN
from ban import BANLayer
from torch.nn.utils.weight_norm import weight_norm
import torch
import torch.nn as nn

class DrugGCN(nn.Module):
    def __init__(self, in_feats, dim_embedding=128, padding=True, hidden_feats=None, activation=None):
        super(DrugGCN, self).__init__()
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        if padding:
            with torch.no_grad():
                self.init_transform.weight[-1].fill_(0)
        # self.gat = SchNetGNN(node_feats=dim_embedding, hidden_feats=hidden_feats)
        # self.gat = GraphSAGE(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=activation)
        # self.gat = GAT(in_feats=dim_embedding, hidden_feats=hidden_feats, activations=activation)
        # self.fpgnn = AttentiveFPGNN(dim_embedding, 13,  num_layers=2, graph_feat_size=128)
        self.gnn = GCN(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=activation)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.output_feats = hidden_feats[-1]

    def forward(self, batch_graph, node_name):
        node_feats = batch_graph.ndata.pop(node_name)
        node_feats = self.init_transform(node_feats)
        node_feats = self.gnn(batch_graph, node_feats)
        batch_size = batch_graph.batch_size
        node_feats = node_feats.view(batch_size, -1, self.output_feats)
        return node_feats


class ProtGCN(nn.Module):
    def __init__(self, in_feats, dim_embedding=128, padding=True, hidden_feats=None, activation=None):
        super(ProtGCN, self).__init__()
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        if padding:
            with torch.no_grad():
                self.init_transform.weight[-1].fill_(0)
        # self.gat = SchNetGNN(node_feats=dim_embedding, hidden_feats=hidden_feats)
        # self.gat = GraphSAGE(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=activation)
        # self.gat = GAT(in_feats=dim_embedding, hidden_feats=hidden_feats, activations=activation)
        # self.fpgnn = AttentiveFPGNN(dim_embedding, 13,  num_layers=2, graph_feat_size=128)
        self.gnn = GCN(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=activation)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.output_feats = hidden_feats[-1]

    def forward(self, batch_graph, node_name):
        node_feats = batch_graph.ndata.pop(node_name)
        node_feats = self.init_transform(node_feats)
        node_feats = self.gnn(batch_graph, node_feats)
        batch_size = batch_graph.batch_size
        node_feats = node_feats.view(batch_size, -1, self.output_feats)
        return node_feats


class CNNBlock(nn.Module):
    def __init__(self, embedding_dim, num_filters, kernel_size, downsample=False):
        super(CNNBlock, self).__init__()
        in_ch = [embedding_dim] + num_filters
        self.in_ch = in_ch[-1]
        kernels = kernel_size
        self.stride = 2 if downsample else 1
        self.layer = nn.Sequential(
            nn.Conv1d(in_channels=in_ch[0], out_channels=in_ch[1], kernel_size=kernels[0], padding=kernels[0]//2, stride=self.stride),
            nn.BatchNorm1d(in_ch[1]),
            nn.ReLU(),

            nn.Conv1d(in_channels=in_ch[1], out_channels=in_ch[2], kernel_size=kernels[1], padding=kernels[1]//2),
            nn.BatchNorm1d(in_ch[2]),
            nn.ReLU(),

            nn.Conv1d(in_channels=in_ch[2], out_channels=in_ch[3], kernel_size=kernels[2], padding=kernels[2]//2),
            nn.BatchNorm1d(in_ch[3]),
            nn.ReLU()
        )

        if in_ch[0] != in_ch[3] or downsample:
            self.res_layer = nn.Conv1d(in_channels=in_ch[0], out_channels=in_ch[3], kernel_size=1, stride=self.stride)
        else:
            self.res_layer = None

    def forward(self, v):
        v = v.transpose(2, 1)    
        residual = v if self.res_layer is None else self.res_layer(v)
        v = self.layer(v) 
        v = v + residual
        v = F.relu(v)
        v = v.transpose(2, 1)
        return v


class MLPBlock(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPBlock, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.bn1(F.dropout(F.relu(self.fc1(x)), 0.3))
        x = self.bn2(F.dropout(F.relu(self.fc2(x)), 0.3))
        x = self.fc3(x)
        return x


class DTIPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.drug_encoder = DrugGCN(in_feats=75, dim_embedding=128, hidden_feats=[128, 128, 128])
        self.prot_encoder = ProtGCN(in_feats=21, dim_embedding=128, hidden_feats=[128, 128, 128])

        self.drug_seq_proj = nn.Linear(768, 128)
        self.prot_seq_proj = nn.Linear(1280, 128)

        self.drug_cnn = CNNBlock(embedding_dim=128, num_filters=[128, 128, 128], kernel_size=[3, 5, 7])
        self.prot_cnn = CNNBlock(embedding_dim=128, num_filters=[128, 128, 128], kernel_size=[3, 5, 7])

        self.inter = weight_norm(
            BANLayer(v_dim=256, q_dim=256, h_dim=256, h_out=2),
            name='h_mat', dim=None)

        self.classifier = MLPBlock(input_size=256, hidden_size=128, output_size=2)

    def forward(self, drug_graph, prot_graph, drug_mask, prot_mask, drug_feat, prot_feat):
        drug_graph_emb = self.drug_encoder(drug_graph, 'node')
        prot_graph_emb = self.prot_encoder(prot_graph, 'node')

        drug_feat = self.drug_seq_proj(drug_feat)  # (B, 100, 128)
        prot_feat = self.prot_seq_proj(prot_feat)  # (B,1500, 128)

        drug_seq_emb = self.drug_cnn(drug_feat)
        prot_seq_emb = self.prot_cnn(prot_feat)

        drug_fused = torch.cat([drug_graph_emb, drug_seq_emb], dim=2)
        prot_fused = torch.cat([prot_graph_emb, prot_seq_emb], dim=2)

        feat, att = self.inter(drug_fused, prot_fused)
        y_pred = self.classifier(feat)
        return att, y_pred