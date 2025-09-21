#!/usr/bin/env python
# Author  : KerryChen
# File    : train_drug.py
# Time    : 2025/7/10 10:25

import copy
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch.nn as nn
from prefetch_generator import BackgroundGenerator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, \
    precision_recall_curve, auc
from tqdm import tqdm
from utils import calculate_metrics
from torch.utils.data import DataLoader
from load_data import LoadMyData, CustomDataset
from model import DTIPredictor
import torch.nn.functional as F
from utils import Logger, EarlyStopping
import dgl
import torch

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def collate_func_train(x):
    dg, pg, dm, pm, df, pf, y = zip(*x)
    batched_dg = dgl.batch(dg)
    batched_pg = dgl.batch(pg)

    drug_mask = torch.stack(dm)
    prot_mask = torch.stack(pm)

    drug_feat = torch.stack(df)      # (B, seq_len_drug, feat_dim)
    prot_feat = torch.stack(pf)      # (B, seq_len_prot, feat_dim)

    y_tensor = torch.tensor(y)

    return batched_dg, batched_pg, drug_mask, prot_mask, drug_feat, prot_feat, y_tensor


def split_train_test(all_set, train_ratio=0.9):
    random.shuffle(all_set)
    total = len(all_set)
    train_end = int(total * train_ratio)
    train_set = all_set[:train_end]
    test_set = all_set[train_end:]
    return train_set, test_set


def split_train_valid_test(all_set, train_ratio=0.8, val_ratio=0.1):
    random.shuffle(all_set)
    total = len(all_set)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    train_set = all_set[:train_end]
    val_set = all_set[train_end:val_end]
    test_set = all_set[val_end:]
    return train_set, val_set, test_set


def calculate_metrics(y_true, y_pred, y_prob):
    """
    Args:
        y_true: list of true labels
        y_pred: list of predicted labels
        y_prob: list of predicted probabilities

    Returns: dict of metrics

    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    # Area Under the Receiver Operation Characteristics curve (AUROC)
    auroc = roc_auc_score(y_true, y_prob)

    # Area Under the Precision-Recall Curve (AUPRC)
    tpr, fpr, _ = precision_recall_curve(y_true, y_prob)
    aupr = auc(fpr, tpr)

    return {'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'MCC': mcc,
            'AUROC': auroc,
            'AUPR': aupr}

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    train_pbar = tqdm(
        enumerate(BackgroundGenerator(loader)),
        total=len(loader),
        desc='Training'
    )

    for train_i, train_data in train_pbar:
        v_dg, v_pg, v_dm, v_pm, v_df, v_pf, labels = train_data
        v_dg = v_dg.to(device)
        v_pg = v_pg.to(device)
        v_dm = v_dm.to(device)
        v_pm = v_pm.to(device)
        v_df = v_df.to(device)
        v_pf = v_pf.to(device)
        labels = labels.long().to(device)

        optimizer.zero_grad()
        _, outputs = model(v_dg, v_pg, v_dm, v_pm, v_df, v_pf)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    total_loss = 0

    test_pbar = tqdm(enumerate(BackgroundGenerator(loader)), total=len(loader))

    for test_i, test_data in test_pbar:
        v_dg, v_pg, v_dm, v_pm, v_df, v_pf, labels = test_data
        v_dg = v_dg.to(device)
        v_pg = v_pg.to(device)
        v_dm = v_dm.to(device)
        v_pm = v_pm.to(device)
        v_df = v_df.to(device)
        v_pf = v_pf.to(device)
        labels = labels.long().to(device)

        _, outputs = model(v_dg, v_pg, v_dm, v_pm, v_df, v_pf)
        loss = criterion(outputs, labels)

        total_loss += loss.item()

        probs = F.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        prob = probs[:, 1]

        y_true.extend(labels.tolist())
        y_pred.extend(preds.tolist())
        y_prob.extend(prob.tolist())

    metrics = calculate_metrics(y_true, y_pred, y_prob)
    avg_loss = total_loss / len(loader)
    return avg_loss, metrics


@torch.no_grad()
def test(model, loader, criterion, device):
    return evaluate(model, loader, criterion, device)


def train_one_seed(cv_root, seed, config, model_dir):

    print(f"\n{'=' * 60}\nTraining Seed {seed}\n{'=' * 60}")

    prefix = config['data_prefix']
    train_file = os.path.join(cv_root, f"seed_{seed}", f"{prefix}_train.csv")
    val_file = os.path.join(cv_root, f"seed_{seed}", f"{prefix}_val.csv")
    test_file = os.path.join(cv_root, f"seed_{seed}", f"{prefix}_test.csv")

    train_set = LoadMyData(train_file)
    valid_set = LoadMyData(val_file)
    test_set = LoadMyData(test_file)

    train_dataset = CustomDataset(train_set)
    valid_dataset = CustomDataset(valid_set)
    test_dataset = CustomDataset(test_set)

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=config['batch_size'],
        drop_last=True,
        collate_fn=collate_func_train,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        shuffle=False,
        batch_size=config['batch_size'],
        drop_last=False,
        collate_fn=collate_func_train,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=config['batch_size'],
        drop_last=False,
        collate_fn=collate_func_train,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    model = DTIPredictor().to(config['device'])
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1, reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5,
                                                           threshold=0.0001, threshold_mode='rel', cooldown=0,
                                                           min_lr=1e-6, eps=1e-08, verbose=False)

    early_stopper = EarlyStopping(patience=config['patience'], delta=1e-4)

    best_score = 0
    start_time = time.time()

    for epoch in range(config['epochs']):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, config['device'])

        valid_loss, val_metrics = evaluate(model, valid_loader, criterion, config['device'])

        scheduler.step(valid_loss)

        val_acc = val_metrics['Accuracy']
        val_pre = val_metrics['Precision']
        val_rec = val_metrics['Recall']
        val_f1 = val_metrics['F1']
        val_mcc = val_metrics['MCC']
        val_roc = val_metrics['AUROC']
        val_pr = val_metrics['AUPR']
        dur = time.time() - start_time

        print(
            f"[Epoch {epoch + 1}/{config['epochs']}] Train Loss: {train_loss:.4f} Valid Loss: {valid_loss:.4f} | "
            f"Acc: {val_acc:.4f} | Prec: {val_pre:.4f} | Recall: {val_rec:.4f} | F1: {val_f1:.4f} | "
            f"MCC: {val_mcc:.4f} | AUROC: {val_roc:.4f} | AUPR: {val_pr:.4f}"
        )

        if val_roc > best_score:
            best_epoch = epoch + 1
            best_score = val_roc
            best_model = copy.deepcopy(model)

        early_stopper.step(val_roc)
        if early_stopper.early_stop:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    torch.save(best_model.state_dict(),
               os.path.join(model_dir, f'{config["data_prefix"]}_{config["task_lower"]}_seed_{seed}.pt'))

    model = best_model
    test_loss, test_metrics = test(model, test_loader, criterion, config['device'])

    print(f"Best model at epoch {best_epoch} with Score = {best_score:.4f}")
    print(f"Seed {seed} Test Results:")
    print(
        f"[Test] Loss: {test_loss:.4f} | Acc: {test_metrics['Accuracy']:.4f} | Prec: {test_metrics['Precision']:.4f} | "
        f"Recall: {test_metrics['Recall']:.4f} | F1: {test_metrics['F1']:.4f} | MCC: {test_metrics['MCC']:.4f} | "
        f"AUROC: {test_metrics['AUROC']:.4f} | AUPR: {test_metrics['AUPR']:.4f}"
    )

    return test_metrics


if __name__ == '__main__':
    task = "SNAP"   #SNAP/DRH/Kinase
    task_lower = task.lower()

    config = {
        'device': torch.device('cuda:2' if torch.cuda.is_available() else 'cpu'),
        'seed': 3407,
        'batch_size': 32,
        'epochs': 100,
        'num_workers': 4,
        'learning_rate': 3e-4,
        'weight_decay': 5e-5,
        'patience': 10,
        'task': task,
        'task_lower': task_lower,
        'data_prefix': 'drug'}

    setup_seed(config['seed'])
    # =========================================================================================
    log_dir = f'../logs/{task}/'
    os.makedirs(log_dir, exist_ok=True)
    model_dir = f'../models/{task}/'
    os.makedirs(model_dir, exist_ok=True)
    result_dir = f'../results/{task}/'
    os.makedirs(result_dir, exist_ok=True)

    data_dir = f'../datasets/{config['task']}/{config["data_prefix"]}_split_results/'

    # =========================================================================================
    log_file = os.path.join(log_dir, f'{config["data_prefix"]}_{config["task_lower"]}_prefix.log')
    sys.stdout = Logger(log_file)
    sys.stdout.log_config(config)

    # =========================================================================================
    seed_results = []
    seeds = [42, 1234, 3407, 5678, 8888]
    for seed in seeds:
        metrics = train_one_seed(data_dir, seed=seed, config=config, model_dir=model_dir)
        seed_results.append({**{'Seed': seed}, **metrics})

    # Convert to DataFrame for better visualization
    df = pd.DataFrame(seed_results).set_index("Seed").round(3)
    mean_std_row = {col: f"{df[col].mean():.3f}±{df[col].std():.3f}" for col in df.columns}
    df.loc["Mean±Std"] = mean_std_row

    print("\n=== Final Test Results (with Mean ± Std) ===")
    print(df)

    out_file = os.path.join(result_dir, f'{config["data_prefix"]}_{task_lower}_results.csv')
    df.to_csv(out_file)
