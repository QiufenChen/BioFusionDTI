import torch.nn as nn
import argparse
import torch
import esm
import pandas as pd
import numpy as np
from rdkit import Chem
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm.auto import tqdm
from models import GraphBAN
from time import time
from utils import set_seed, graph_collate_func, mkdir,graph_collate_func2
from configs import get_cfg_defaults
from dataloader import DTIDataset, MultiDataLoader,DTIDataset2
from trainer import Trainer
from domain_adaptator import Discriminator
import argparse
import warnings, os
import pandas as pd
import copy
from rdkit.Chem import AllChem
import math
from dgllife.model.gnn import GCN
from torch.nn.utils.weight_norm import weight_norm
import pickle
import random
import time


# Set up argument parser
parser = argparse.ArgumentParser(description="Load train, val, test datasets and additional parameters.")

parser.add_argument("--train_path", type=str, required=True, help="Path to the train dataset.")
parser.add_argument("--val_path", type=str, required=True, help="Path to the validation dataset.")
parser.add_argument("--test_path", type=str, required=True, help="Path to the test dataset.")
parser.add_argument("--seed", type=int, required=True, help="Seed number for random processes.")
parser.add_argument("--mode", type=str, choices=['inductive', 'transductive'], required=True, help="Mode of operation: 'inductive' or 'transductive'.")
parser.add_argument("--teacher_path", type=str, required=True, help="Path to the teacher Parquet file.")
parser.add_argument("--save_dir", type=str, required=True, help="Path to save file.")
# Parse the command line arguments
args = parser.parse_args()



def count_atoms(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return mol.GetNumAtoms()

def filter_large_molecules(df: pd.DataFrame, threshold: int = 290) -> pd.DataFrame:
    """
    在 DataFrame 中筛选 SMILES 原子数大于 threshold 的分子
    """
    df = df.copy()
    df["num_atoms"] = df["SMILES"].apply(count_atoms)
    return df[df["num_atoms"] < threshold]

# Load the datasets using the provided paths
_df_train = pd.read_csv(args.train_path)
_df_val = pd.read_csv(args.val_path)
_df_test = pd.read_csv(args.test_path)

df_train = filter_large_molecules(_df_train, threshold=290)
df_val = filter_large_molecules(_df_val, threshold=290)
df_test = filter_large_molecules(_df_test, threshold=290)

 
# Print the shapes of the datasets
print(df_train.shape)
print(df_val.shape)
print(df_test.shape)
print("Seed: ", args.seed)
print("Mode: ", args.mode)
#print("Teacher data shape: ", df_teacher.shape)

df_test['SEQUENCE'] = df_test['SEQUENCE'].apply(lambda x: x[:1022] if len(x) > 1022 else x)
df_train['SEQUENCE'] = df_train['SEQUENCE'].apply(lambda x: x[:1022] if len(x) > 1022 else x)
df_val['SEQUENCE'] = df_val['SEQUENCE'].apply(lambda x: x[:1022] if len(x) > 1022 else x)




device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
esm_model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
batch_converter = alphabet.get_batch_converter()

esm_model = esm_model.eval().to(device)


def Get_Protein_Feature(p_list):
    feature = []
    data_tmp = []
    dictionary = {}
    i = 0
    for p in p_list:
        p = p[0:1022]
        data_tmp.append(("SEQUENCE" + str(i), p))
        i = i + 1
    # print(len(data_tmp))

    sequence_representations = []

    for i in tqdm(range(len(data_tmp) // 5 + 1)):
        # print(i)
        if i == len(data_tmp) // 5:
            data_part = data_tmp[i * 5:]
        else:
            data_part = data_tmp[i * 5:(i + 1) * 5]

        if not data_part:  # Check if data_part is empty
            continue

        data_part = [(label, sequence) for label, sequence in data_part]
        _, _, batch_tokens = batch_converter(data_part)
        batch_tokens = batch_tokens.to(device)
        with torch.no_grad():
            results = esm_model(batch_tokens, repr_layers=[33], return_contacts=False)
        token_representations = results["representations"][33]

        for j, (_, seq) in enumerate(data_part):
            emb_rep = token_representations[j, 1:len(seq) + 1].mean(0)
            emb_rep = emb_rep.cpu().numpy()
            # sequence_representations.append(emb_rep.cpu().numpy())
            dictionary[seq] = emb_rep
            df = pd.DataFrame(dictionary.items(), columns=['SEQUENCE', 'esm'])
            # dictionary[seq] = token_representations[j, 1 : len(seq) + 1].mean(0)
    # np.save('biosnap_protein_feature.npy', dictionary)
    # print(len(sequence_representations))

    return df


pro_list_train = df_train['SEQUENCE'].unique()
x_train = Get_Protein_Feature(list(pro_list_train))
df_train = pd.merge(df_train, x_train, on='SEQUENCE', how='left')
print('train esm is done!\n')

pro_list_val = df_val['SEQUENCE'].unique()
x_val = Get_Protein_Feature(list(pro_list_val))
df_val = pd.merge(df_val, x_val, on='SEQUENCE', how='left')
print('val esm is done!\n')


pro_list_test = df_test['SEQUENCE'].unique()
x = Get_Protein_Feature(list(pro_list_test))
df_test = pd.merge(df_test, x, on='SEQUENCE', how='left')
print('test esm is done!\n')


print('ESM feature extraction: pass')
from transformers import AutoTokenizer, AutoModelForSequenceClassification, RobertaModel
#from transformers import TrainingArguments, Trainer, IntervalStrategy

# Setup
# Load a pretrained transformer model and tokenizer
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = "DeepChem/ChemBERTa-77M-MTR"
chem_path = r"./chemberta_model"
token_path = r"./tokenizer"
model_chem = RobertaModel.from_pretrained(chem_path, num_labels=2, add_pooling_layer=True, use_safetensors=True)
tokenizer = AutoTokenizer.from_pretrained(token_path)
model_chem = model_chem.to(device)

def get_embeddings(df):
    emblist = []
    #embedding_df = pd.DataFrame(columns=['SMILES'] + [f'chemberta2_feature_{i}' for i in range(1, 385)])

    for index, row in tqdm(df.iterrows(), total=df.shape[0], leave=False):
      # truncate to the maximum length accepted by the model if no max_length is provided
        encodings = tokenizer(row['SMILES'], return_tensors='pt',padding="max_length", max_length=290, truncation=True)
        encodings = encodings.to(device)
        with torch.no_grad():
            output = model_chem(**encodings)
            smiles_embeddings = output.last_hidden_state[0, 0, :]
            #smiles_embeddings = smiles_embeddings.squeeze(0)
            smiles_embeddings = smiles_embeddings.cpu()
            smiles_embeddings = smiles_embeddings.detach().numpy().astype(np.float64)

            emblist.append(smiles_embeddings)

        # Ensure you move the tensor back to cpu for numpy conversion
        #dic = {**{'SMILES': row['SMILES']}, **dict(zip([f'chemberta2_feature_{i}' for i in range(1, 385)], smiles_embeddings.cpu().numpy().tolist()))}
        #embedding_df.loc[len(embedding_df)] = pd.Series(dic)

    return emblist#smiles_embeddings

df_trainu = df_train.drop_duplicates(subset='SMILES').reset_index(drop=True)
df_valu = df_val.drop_duplicates(subset='SMILES').reset_index(drop=True)
df_testu = df_test.drop_duplicates(subset='SMILES').reset_index(drop=True)

emblist_train = get_embeddings(df_trainu)
df_trainu['fcfp'] = emblist_train

emblist_val = get_embeddings(df_valu)
df_valu['fcfp'] = emblist_val

emblist_test = get_embeddings(df_testu)
df_testu['fcfp'] = emblist_test

# Merge DataFrames on 'SMILES' column
df_train = pd.merge(df_train, df_trainu[['SMILES', 'fcfp']], on='SMILES', how='left').reset_index(drop=True)
df_val = pd.merge(df_val, df_valu[['SMILES', 'fcfp']], on='SMILES', how='left').reset_index(drop=True)
df_test = pd.merge(df_test, df_testu[['SMILES', 'fcfp']], on='SMILES', how='left').reset_index(drop=True)
print('chemBERTa feature extraction: pass\n')


# If you want to change the settings such as number of epochs for teh GraphBANs main model change it through GraphBAN_Demo.yaml.
# If you want to run the model for transductive analysis, use GraphBAN_None_DA.yaml
if args.mode == 'inductive':
    cfg_path = "./inductive_mode/GraphBAN_DA.yaml"
else:
    cfg_path = "./inductive_mode/GraphBAN.yaml"


cfg = get_cfg_defaults()
cfg.merge_from_file(cfg_path)
cfg.freeze()
torch.cuda.empty_cache()
warnings.filterwarnings("ignore")
set_seed(args.seed)

save_dir = args.save_dir
mkdir(save_dir)

# mkdir(cfg.RESULT.OUTPUT_DIR)
experiment = None
print(f"Config yaml: {cfg_path}")
print(f"Running on: {device}")
print(f"Hyperparameters:")
print(dict(cfg))


train_emb = pd.read_parquet(args.teacher_path)
train_emb['Array'] = train_emb.apply(lambda row: np.array(row), axis=1)

# Drop all columns except the 'Array' column
train_emb.drop(train_emb.columns.difference(['Array']), axis=1, inplace=True)

df_train['teacher_emb'] = train_emb['Array']

train_dataset = DTIDataset2(df_train.index.values, df_train)
val_dataset = DTIDataset(df_val.index.values, df_val)
test_dataset = DTIDataset(df_test.index.values, df_test)

params1 = {'batch_size': cfg.SOLVER.BATCH_SIZE, 'shuffle': True, 'num_workers': cfg.SOLVER.NUM_WORKERS, 'drop_last': True, 'collate_fn': graph_collate_func}
params2 = {'batch_size': cfg.SOLVER.BATCH_SIZE, 'shuffle': True, 'num_workers': cfg.SOLVER.NUM_WORKERS, 'drop_last': True, 'collate_fn': graph_collate_func2}
source_generator = DataLoader(train_dataset, **params2)
target_generator = DataLoader(val_dataset, **params1)
n_batches = max(len(source_generator), len(target_generator))
multi_generator = MultiDataLoader(dataloaders=[source_generator, target_generator], n_batches=n_batches)
training_generator = DataLoader(train_dataset, **params2)
params1['shuffle'] = False
params1['drop_last'] = False
if args.mode == 'inductive':
    val_generator = DataLoader(test_dataset,**params1)
else:
    val_generator = DataLoader(val_dataset,**params1)

test_generator = DataLoader(test_dataset,**params1)

modelG = GraphBAN(**cfg).to(device)
opt = torch.optim.Adam(modelG.parameters(), lr=cfg.SOLVER.LR)
if torch.cuda.is_available():
  torch.backends.cudnn.benchmark = True


if cfg.DA.USE:
        if cfg["DA"]["RANDOM_LAYER"]:
            domain_dmm = Discriminator(input_size=cfg["DA"]["RANDOM_DIM"], n_class=cfg["DECODER"]["BINARY"]).to(device)
        else:
            domain_dmm = Discriminator(input_size=cfg["DECODER"]["IN_DIM"] * cfg["DECODER"]["BINARY"],
                                       n_class=cfg["DECODER"]["BINARY"]).to(device)
        # params = list(model.parameters()) + list(domain_dmm.parameters())
        opt = torch.optim.Adam(modelG.parameters(), lr=cfg.SOLVER.LR)
        opt_da = torch.optim.Adam(domain_dmm.parameters(), lr=cfg.SOLVER.DA_LR)
else:
        opt = torch.optim.Adam(modelG.parameters(), lr=cfg.SOLVER.LR)
torch.backends.cudnn.benchmark = True

if not cfg.DA.USE:

    trainer = Trainer(modelG, opt, device, training_generator, val_generator, test_generator, save_dir,
                          opt_da=None,
                          discriminator=None,
                          experiment=experiment, **cfg)
else:
    trainer = Trainer(modelG, opt, device, multi_generator, val_generator, test_generator, save_dir,
                         opt_da=opt_da,
                         discriminator=domain_dmm,
                         experiment=None, **cfg)



result = trainer.train()

# random_num = int(time.time() * 1000) % 10000
# pickle.dump(result, open(os.path.join(r"/home/qfchen/GraphBAN/result", f'test_output_{random_num}.pkl'),'wb'))
# print('pass')
