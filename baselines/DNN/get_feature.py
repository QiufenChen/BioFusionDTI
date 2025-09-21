import os
import pickle
import time

import numpy as np
import pandas as pd
from propy import PyPro
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator


def get_morgan_fp(mol):
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
    _, file_extension = os.path.splitext(file_path)

    if file_extension.lower() == '.csv':
        df = pd.read_csv(file_path)
        return df
    elif file_extension.lower() == '.xlsx':
        df = pd.read_excel(file_path, engine="openpyxl")
        return df
    else:
        raise ValueError(f"Unsupported file types: {file_extension}")


def LoadMyData(my_file, prefix):
    """
    Load dataset from a file and save as .npy.
    Args:
        my_file (str): Path to the dataset file.
        save_path (str): Path to save the output .npy file.
    Returns:
        np.ndarray: The dataset as a NumPy array.
    """
    start_time = time.time()
    print(f"Loading dataset from {my_file}...", flush=True)

    save_dir = './data/'
    os.makedirs(save_dir, exist_ok=True)

    # Read the input file
    df = read_file(my_file)
    print("Columns in the dataset:", df.columns.tolist())

    drug_dict = {}
    prot_dict = {}

    unique_drugs = df[['DRUG_ID', 'SMILES']].drop_duplicates()
    for _, row in unique_drugs.iterrows():
        drug_id, smiles = row['DRUG_ID'], row['SMILES']
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"[Drug Skipped] Invalid SMILES: {smiles}")
            continue
        try:
            drug_dict[drug_id] = get_morgan_fp(mol)
        except Exception as e:
            print(f"[Drug Error] {drug_id}: {e}")

    with open(os.path.join(save_dir, f'{prefix}_drug_features.pkl'), 'wb') as f:
        pickle.dump(drug_dict, f)
    print(f"Saved drug features to {os.path.join(save_dir, f'{prefix}_drug_features.pkl')}")


    unique_prots = df[['UNIPROT_ID', 'SEQUENCE']].drop_duplicates()
    for _, row in unique_prots.iterrows():
        prot_id, seq = row['UNIPROT_ID'], row['SEQUENCE'].upper()
        if any(x in seq for x in ['X', 'U', 'O']):
            print(f"[Protein Skipped] Non-standard amino acids in {prot_id}")
            continue
        try:
            prot_dict[prot_id] = get_physic_chemical_propertity(seq)
        except Exception as e:
            print(f"[Protein Error] {prot_id}: {e}")


    with open(os.path.join(save_dir, f'{prefix}_protein_features.pkl'), 'wb') as f:
        pickle.dump(prot_dict, f)
    print(f"Saved protein features to {os.path.join(save_dir, f'{prefix}_protein_features.pkl')}")


if __name__ == "__main__":
    data_name = 'SNAP'  #SNAP, DRH, Kinase
    data_name_lower = data_name.lower()
    input_file = f'../datasets/{data_name}/{data_name}.xlsx'
    data = LoadMyData(input_file, data_name_lower)

