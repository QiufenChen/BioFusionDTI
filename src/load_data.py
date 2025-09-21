#!/usr/bin/env python
# Author  : KerryChen
# File    : load_data.py
# Time    : 2024/10/12 16:02
import os
import pickle

import dgl
from rdkit import Chem
import pandas as pd
import numpy as np
import time
# from get_features import extract_drug_features
from torch.utils.data import Dataset
import torch
from functools import partial
from dgl import add_self_loop
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
import warnings
warnings.filterwarnings('ignore')



def get_node_mask(graph, feat_key='feat'):
    node_feats = graph.ndata[feat_key]  # shape: [num_nodes, feature_dim]
    is_virtual = node_feats[:, -1] == 1  # shape: [num_nodes]
    mask = ~is_virtual  # True for real nodes, False for virtual nodes
    return mask  # shape: [num_nodes], dtype: torch.bool


def parsePSSM(uniprot_id):
    pssm_file = f'../pssm/{uniprot_id}.pssm'
    pssm_info = open(pssm_file)
    lines = pssm_info.readlines()

    idxs = []
    pssm = []
    headers = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]

    special_chars = {'U', 'Z', 'O', 'B'}

    for line in lines:
        cols = line.strip().split()

        if len(cols) == 44:
            residue = cols[1]
            if residue in special_chars:
                residue = 'X'
            idxs.append(residue)
            scores = list(map(float, cols[2:22]))
            pssm.append(scores)

    pssm_df = pd.DataFrame(pssm, columns=headers)
    pssm_df.insert(loc=0, column='residue_letter', value=idxs)
    return pssm_df


def build_drug_graph(sml, max_drug_nodes=100):
    atom_featurizer = CanonicalAtomFeaturizer()
    bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
    custom_func = partial(smiles_to_bigraph, add_self_loop=True)

    graph = custom_func(smiles=sml, node_featurizer=atom_featurizer, edge_featurizer=bond_featurizer)

    actual_node_feats = graph.ndata.pop('h')
    num_actual_nodes = actual_node_feats.shape[0]
    num_virtual_nodes = max(0, max_drug_nodes - num_actual_nodes)

    virtual_node_bit = torch.zeros([num_actual_nodes, 1])
    actual_node_feats = torch.cat((actual_node_feats, virtual_node_bit), dim=1)
    graph.ndata['node'] = actual_node_feats

    virtual_node_feat = torch.cat((torch.zeros(num_virtual_nodes, 74), torch.ones(num_virtual_nodes, 1)), dim=1)
    graph.add_nodes(num_virtual_nodes, {'node': virtual_node_feat})

    graph = add_self_loop(graph)
    return graph


def build_protein_graph(uniprot_id, max_protein_nodes=1500):
    """PSSM"""
    pssm_df = parsePSSM(uniprot_id)
    seq_features = pssm_df.iloc[:, -20:].values

    # """Onehot"""
    # embedding_layer = nn.Embedding(num_embeddings=26, embedding_dim=128, padding_idx=0)
    # seq_features = get_one_hot(sequence, embedding_layer)
    # """ProtBert"""
    # bert_file = f'/home/qfchen/esm2_feature/{uniprot_id}.pt'
    # seq_features = torch.load(bert_file)
    feature_dim = seq_features.shape[1]

    contact_file = f'../esm2_contact/{uniprot_id}.pt'
    contact_map = torch.load(contact_file).numpy()
    # print(contact_map.shape, seq_features.shape)
    if len(contact_map) > max_protein_nodes:
        contact_map = contact_map[:max_protein_nodes, :max_protein_nodes]
        seq_features = seq_features[:max_protein_nodes]
        num_nodes = max_protein_nodes
    else:
        num_nodes = len(contact_map)

    src, dst = np.where((contact_map > 0.5))
    graph = dgl.graph((src, dst), num_nodes=num_nodes)

    real_feat = torch.cat([torch.tensor(seq_features, dtype=torch.float32), torch.zeros((num_nodes, 1))], dim=1)
    graph.ndata['node'] = real_feat

    edge_weights = contact_map[src, dst]
    graph.edata['edge'] = torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(1)

    num_virtual_nodes = max(0, max_protein_nodes - num_nodes)
    if num_virtual_nodes > 0:
        virtual_feat = torch.cat([torch.zeros((num_virtual_nodes, feature_dim)), torch.ones((num_virtual_nodes, 1))], dim=1)
        graph.add_nodes(num_virtual_nodes, {'node': virtual_feat})

    graph = dgl.add_self_loop(graph)
    return graph


def ParseSequence(sequence, max_seq_len):
    amino_acids = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
                   "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
                   "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
                   "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}

    protein_indices = np.zeros(max_seq_len, dtype=np.int64())
    for i, ch in enumerate(sequence[:max_seq_len]):
        protein_indices[i] = amino_acids[ch]
    return protein_indices


def get_one_hot(sequence, embedding_layer):
    amino_acids = {
        "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
        "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
        "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
        "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25
    }

    indices = np.zeros(len(sequence), dtype=np.int64)
    for i, ch in enumerate(sequence):
        indices[i] = amino_acids.get(ch, 0) 

    indices_tensor = torch.LongTensor(indices).unsqueeze(0)  # shape: [1, seq_len]
    embedded = embedding_layer(indices_tensor)  # shape: [1, seq_len, 128]
    return embedded.squeeze(0)  # shape: [seq_len, 128]


def ParseSMILES(smi, max_smi_len):
    smi_chars = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64,
                 "p":65, "*":66}

    drug_indices = np.zeros(max_smi_len, dtype=np.int64())
    for i, ch in enumerate(smi[:max_smi_len]):
        drug_indices[i] = smi_chars[ch]
    return drug_indices


def sequence_features(uniprot_id, max_seq_len=1500) -> torch.Tensor:
    # bert_file = f'/home/qfchen/protbert_feature/{uniprot_id}.pt'
    # bert_file = f'/home/qfchen/protbert_bfd_feature/{uniprot_id}.pt'
    bert_file = f'../esm2_feature/{uniprot_id}.pt'
    if os.path.exists(bert_file):
        seq_feat = torch.load(bert_file)
        if seq_feat.shape[0] > max_seq_len:
            final_seq_feat = seq_feat[:max_seq_len, :]
        else:
            padding_length = max_seq_len - seq_feat.shape[0]
            padding = torch.zeros((padding_length, seq_feat.shape[1]), dtype=seq_feat.dtype)
            final_seq_feat = torch.cat([seq_feat, padding], dim=0)
        return final_seq_feat


def read_file(file_path):
    _, file_extension = os.path.splitext(file_path)
    if file_extension.lower() == '.csv':
        df = pd.read_csv(file_path)
        return df
    elif file_extension.lower() == '.xlsx':
        df = pd.read_excel(file_path, engine="openpyxl")
        return df
    elif file_extension.lower() == '.pkl':
        df = pd.read_pickle(file_path)
        return df
    else:
        raise ValueError(f"Unsupported file types: {file_extension}")


def count_atoms(smiles: str) -> int:
    """Count the number of atoms in a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:  # Invalid SMILES
        return 0
    return mol.GetNumAtoms()


def LoadMyData(my_file, labeled=True):
    t0 = time.time()
    print(f"Loading data {my_file}......", flush=True)

    pairs_df = read_file(my_file)
    print("Columns in the dataset:", pairs_df.columns.tolist())


    with open('../chembert_feature/snap_drug_feat.pkl', 'rb') as f:
        drug_features = pickle.load(f)

    pair_data = []
    for idx, row in pairs_df.iterrows():
        sml = row['SMILES']
        seq = row['SEQUENCE'].upper()
        prot_id = row['UNIPROT_ID']
        drug_id = row['DRUG_ID']
        label = row['Label'] if labeled else idx

        drug_feat = drug_features[drug_id]
        prot_feat = sequence_features(prot_id)

        mol = Chem.MolFromSmiles(sml)
        if mol is None: continue
        if count_atoms(sml) > 100: continue

        try:
            drug_graph = build_drug_graph(sml)
            prot_graph = build_protein_graph(prot_id)
            drug_mask = get_node_mask(drug_graph, feat_key='node')
            prot_mask = get_node_mask(prot_graph, feat_key='node')

        except Exception as e:
            print(f"{prot_id}/{drug_id}: {str(e)}")
            continue

        pair_data.append((drug_graph, prot_graph, drug_mask, prot_mask, drug_feat, prot_feat, label))

    print(f"Loaded {len(pair_data)} valid pairs (time: {time.time() - t0:.2f}s)")
    return pair_data


class CustomDataset(Dataset):
    def __init__(self, pair_data):
        self.data = pair_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        drug_graph = self.data[idx][0]
        prot_graph = self.data[idx][1]

        drug_mask = self.data[idx][2]
        prot_mask = self.data[idx][3]

        drug_feat = self.data[idx][4]
        prot_feat = self.data[idx][5]

        label = torch.tensor(self.data[idx][6])
        return drug_graph, prot_graph, drug_mask, prot_mask, drug_feat, prot_feat, label


# class MultiDataLoader(object):
#     def __init__(self, dataloaders, n_batches):
#         if n_batches <= 0:
#             raise ValueError("n_batches should be > 0")
#         self._dataloaders = dataloaders
#         self._n_batches = np.maximum(1, n_batches)
#         self._init_iterators()

#     def _init_iterators(self):
#         self._iterators = [iter(dl) for dl in self._dataloaders]

#     def _get_nexts(self):
#         def _get_next_dl_batch(di, dl):
#             try:
#                 batch = next(dl)
#             except StopIteration:
#                 new_dl = iter(self._dataloaders[di])
#                 self._iterators[di] = new_dl
#                 batch = next(new_dl)
#             return batch

#         return [_get_next_dl_batch(di, dl) for di, dl in enumerate(self._iterators)]

#     def __iter__(self):
#         for _ in range(self._n_batches):
#             yield self._get_nexts()
#         self._init_iterators()

#     def __len__(self):
#         return self._n_batches
