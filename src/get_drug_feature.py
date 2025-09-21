#!/usr/bin/env python
# Author  : KerryChen
# File    : get_drug_feature.py
# Time    : 2025/7/26 15:51

import os
import pickle

import pandas as pd
import torch
from tqdm import tqdm
from rdkit import Chem
from transformers import AutoModel, RobertaTokenizer


def extract_drug_features(task, drug_df, model_path: str, device: str = "cpu"):
    """
    Extract molecular features using ChemBERTa and return as a dictionary.

    Args:
        drug_df (pd.DataFrame): Must contain 'DRUG_ID' and 'SMILES' columns.
        model_path (str): Path to the pre-trained ChemBERTa model.
        device (str): Device to run the model, e.g., "cuda" or "cpu".

    Returns:
        dict: {drug_id: feature_tensor [seq_len, hidden_dim]}
    """
    mol_tokenizer = RobertaTokenizer.from_pretrained(model_path)
    mol_encoder = AutoModel.from_pretrained(model_path).to(device)
    mol_encoder.eval()

    drug_feat = {}
    for _, row in tqdm(drug_df.iterrows(), total=len(drug_df)):
        drug_id = row['DRUG_ID']
        smiles = row['SMILES']
        tokens = mol_tokenizer(
            smiles,
            max_length=100,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(device)

        with torch.no_grad():
            out = mol_encoder(**tokens)

        feat = out.last_hidden_state.detach().cpu()[0]  # shape: [seq_len, hidden_dim]
        drug_feat[drug_id] = feat
    
    save_dir = '../chembert_feature/'
    os.makedirs(save_dir, exist_ok=True)

    with open(save_dir + f"{task}_drug_feat.pkl", "wb") as f:
        pickle.dump(drug_feat, f)


if __name__ == '__main__':
    task = 'SNAP'
    task_lower = task.lower()

    pair_df = pd.read_excel(f'../datasets/{task}/{task}.xlsx')

    drug_df = pair_df[['DRUG_ID', 'SMILES']].drop_duplicates()
    drug_features = extract_drug_features(
        task_lower,
        drug_df,
        model_path="../lager_model/ChemBERTa-zinc-base-v1",
        device="cpu"
    )