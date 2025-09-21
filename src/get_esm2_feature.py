#!/usr/bin/env python
# Author  : KerryChen
# File    : get_esm_feature.py
# Time    : 2025/8/11 16:18

import os
import numpy as np
import pandas as pd
import esm
import torch
from tqdm import tqdm


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
    

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
save_dir = '../esm2_feature/'
os.makedirs(save_dir, exist_ok=True)

task = 'SNAP'   # SNAP/DRH/Kinase
DATASET_PATH = f'../datasets/{task}/{task}.xlsx'
df = read_file(DATASET_PATH)
print(df.columns)
assert 'SEQUENCE' in df.columns, "Contains a 'SEQUENCE' column."

model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model = model.to(device)
model.eval()
batch_converter = alphabet.get_batch_converter()

# max_len = 2000
for _, row in tqdm(df[['UNIPROT_ID', 'SEQUENCE']].drop_duplicates().iterrows(), total=len(df)):
    pid = row['UNIPROT_ID']
    seq = row['SEQUENCE']
    seq_len = len(seq)

    # if seq_len > max_len:
    #     seq = seq[:max_len]
    #     seq_len = max_len

    data = [(pid, seq)]
    _, _, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    batch_tokens = batch_tokens.to(device)

    try:
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33])  
            seq_feat = results["representations"][33]       
            
            # Remove the features of [CLS] and [EOS]​
            seq_feat = seq_feat[:, 1: batch_lens - 1] 
            seq_feat = seq_feat.squeeze(dim=0)
            seq_feat = seq_feat.cpu()

            save_path = os.path.join(save_dir, f"{pid}.pt")
            torch.save(seq_feat, save_path)

    except Exception as e:
        print(f"[Error] Failed for {pid}: {e}")