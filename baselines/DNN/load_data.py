'''
# !/usr/bin/python3
# -*- coding: utf-8 -*-
@Time : 2024/3/10 18:40
@Author : Qiufen.Chen
@FileName: utils.py
@Software: PyCharm
'''
import os
import pickle

import numpy as np
import pandas as pd
from propy import PyPro
from rdkit import Chem
import time

from torch.utils.data import Dataset
import torch
import warnings
warnings.filterwarnings('ignore')
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

def get_morgan_fp(mol, radius=3, n_bits=1024):
    """
    ECFP4 (Morgan Fingerprint with radius=2)
    """
    generator = GetMorganGenerator(radius=2, fpSize=1024)
    fp = list(generator.GetFingerprint(mol))
    return np.array(fp)


def get_physic_chemical_propertity(sequence):
    desc = PyPro.GetProDes(sequence)
    aac = desc.GetAAComp()
    pseudo_aac = desc.GetPAAC()

    # 1. Extract the amino acid composition (in the order of standard amino acids)
    amino_acids = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I',
                   'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    aac_values = [aac[aa] for aa in amino_acids]

    # 2. Extract the pseudo-amino acid composition (in the order of PAAC1-PAAC30)
    paac_keys = [f'PAAC{i}' for i in range(1, 31)]
    paac_values = [pseudo_aac[key] for key in paac_keys]
    feature = np.array(aac_values + paac_values, dtype=np.float32)
    return feature


def read_file(file_path):
    _, ext = os.path.splitext(file_path)
    if ext == ".csv":
        return pd.read_csv(file_path)
    elif ext == ".xlsx":
        return pd.read_excel(file_path, engine="openpyxl")
    elif ext == ".pkl":
        return pd.read_pickle(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


# def LoadMyData(my_file):
#     """
#     Load dataset from a file.
#     Args:
#         my_file (str): Path to the dataset file.
#     Returns:
#         list: A list of tuples containing (X, y), X is the feature matrix and y is the label (0/1).
#     """
#     t0 = time.time()
#     print(f"Loading dataset from {my_file}...", flush=True)
#
#     # Read the input file
#     pairs_df = read_file(my_file)
#     print("Columns in the dataset:", pairs_df.columns.tolist())
#
#     pair_data = []
#     for idx, row in pairs_df.iterrows():
#         sml = row['SMILES']
#         seq = row['Sequence'].upper()  # Ensure sequence is uppercase
#
#         label = row['Label']
#
#         # Skip invalid molecules or sequences
#         mol = Chem.MolFromSmiles(sml)
#         if mol is None:
#             print(f"Invalid SMILES: {sml}")
#             continue
#         if 'X' in seq or 'U' in seq or 'O' in seq:
#             print("The sequence contains 'X'")
#             continue
#
#         # Build drug and protein feature
#         try:
#             drug_feat = get_morgan_fp(mol)
#             prot_feat = get_physic_chemical_propertity(seq)
#         except Exception as e:
#             print(f"Error building feature: {str(e)}")
#             continue
#
#         # Append the valid pair
#         pair_data.append([drug_feat, prot_feat, label])
#     print(f"Loaded {len(pair_data)} valid pairs (time: {time.time() - t0:.2f}s)")
#     return np.array(pair_data)

def LoadMyData(my_file, task):
    """
    Load dataset from a file and save as .npy.
    Args:
        my_file (str): Path to the dataset file.
        save_path (str): Path to save the output .npy file.
    Returns:
        np.ndarray: The dataset as a NumPy array.
    """
    t0 = time.time()
    print(f"Loading dataset from {my_file}...", flush=True)

    # Read the input file
    df = read_file(my_file)
    print("Columns in the dataset:", df.columns.tolist())

    with open(f'/home/qfchen/ProteinDrugInter/CompareModel/RF/data/{task}_drug_features.pkl', 'rb') as f:
        drug_dict = pickle.load(f)
        # print(drug_dict)

    with open(f'/home/qfchen/ProteinDrugInter/CompareModel/RF/data/{task}_protein_features.pkl', 'rb') as f:
        prot_dict = pickle.load(f)
        # print(prot_dict)

    pairs = []
    for _, row in df.iterrows():
        drug_id = row['DRUG_ID']
        prot_id = row['UNIPROT_ID']
        # drug_id = row['DRUG_NAME']
        # prot_id = row['UNIPROT_ID']
        label = row['Label']

        if drug_id not in drug_dict or prot_id not in prot_dict:
            # print(drug_id, prot_id, label, flush=True)
            continue

        drug_feat = drug_dict[drug_id]
        prot_feat = prot_dict[prot_id]
        # print(drug_feat.shape, prot_feat.shape)
        pairs.append([drug_feat, prot_feat, label])

    print(f"Loaded {len(pairs)} valid pairs (time: {time.time() - t0:.2f}s)")
    return pairs


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        drug_feat = torch.tensor(self.data[idx][0])
        prot_feat = torch.tensor(self.data[idx][1])
        label = torch.tensor(self.data[idx][2])

        return drug_feat, prot_feat, label


