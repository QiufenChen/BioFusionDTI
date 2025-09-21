#!/usr/bin/env python
# Author  : KerryChen
# File    : get_contact_map.py
# Time    : 2025/7/23 10:54

import pandas as pd
import esm
import os
import torch
import numpy as np
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


device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
save_dir = '../esm2_contact/'
os.makedirs(save_dir, exist_ok=True)

task = 'SNAP'   # SNAP/DRH/Kinase
DATASET_PATH = f'../datasets/{task}/{task}.xlsx'
df = read_file(DATASET_PATH)
assert 'SEQUENCE' in df.columns, "Contains a 'SEQUENCE' column."

model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model = model.to(device)
model.eval()
batch_converter = alphabet.get_batch_converter()

max_len = 1000     
interval = 500   

for _, row in tqdm(df[['UNIPROT_ID', 'SEQUENCE']].drop_duplicates().iterrows(), total=len(df)):
    pid = row['UNIPROT_ID']
    seq = row['SEQUENCE']
    seq_len = len(seq)

    save_path = os.path.join(save_dir, f"{pid}.pt")
    if os.path.exists(save_path):
        print(f"[Skip] {pid}.pt already exists. Skipping...")
        continue

    try:
        if seq_len <= max_len:
            data = [(pid, seq)]
            _, _, batch_tokens = batch_converter(data)
            batch_tokens = batch_tokens.to(device)

            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33], return_contacts=True)
                contact_map = results["contacts"][0].cpu()  # shape: [seq_len, seq_len]

            torch.save(contact_map, save_path)
            del batch_tokens, results
            torch.cuda.empty_cache()

        else:
            # === Long sequences: Extract contact maps in segments and concatenate them ===
            contact_map = np.zeros((seq_len, seq_len), dtype=np.float32)
            count_map = np.zeros((seq_len, seq_len), dtype=np.float32)
            starts = list(range(0, seq_len, interval))

            for start in starts:
                end = min(start + max_len, seq_len)
                sub_seq = seq[start:end]
                data = [(pid, sub_seq)]
                _, _, batch_tokens = batch_converter(data)
                batch_tokens = batch_tokens.to(device)

                with torch.no_grad():
                    results = model(batch_tokens, repr_layers=[33], return_contacts=True)
                    sub_contact = results["contacts"][0].cpu().numpy()  # shape: [L, L]

                sub_len = end - start
                contact_map[start:end, start:end] += sub_contact[:sub_len, :sub_len]
                count_map[start:end, start:end] += 1

                del results, batch_tokens
                torch.cuda.empty_cache()

                if end == seq_len:
                    break

            contact_map = np.divide(contact_map, count_map, out=np.zeros_like(contact_map), where=count_map != 0)
            contact_tensor = torch.from_numpy(contact_map)

            torch.save(contact_tensor, save_path)

    except torch.cuda.OutOfMemoryError:
        print(f"[OOM] Skipping {pid} due to CUDA OOM")
        torch.cuda.empty_cache()
        continue

    except Exception as e:
        print(f"[Error] Failed on {pid}: {e}")
        continue


