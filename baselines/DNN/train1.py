#!/usr/bin/env python
# Author  : KerryChen
# File    : train1.py
# Time    : 2025/7/3 11:07


import copy
import random
import sys
import time

import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             matthews_corrcoef, roc_auc_score, precision_recall_curve, auc)
from torch.utils.data import DataLoader
from utils import Logger, EarlyStopping
from load_data import LoadMyData, CustomDataset
import torch.nn.functional as F
from model import DNN
import os
from prefetch_generator import BackgroundGenerator
from tqdm import tqdm


#%%
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


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
            'AUROC': auroc,
            'AUPR': aupr}


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for drug_feat, prot_feat, labels in train_loader:
        drug_feat, prot_feat, labels = drug_feat.to(device), prot_feat.to(device), labels.long().to(device)

        optimizer.zero_grad()
        outputs = model(drug_feat, prot_feat)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    return avg_loss


def validate(model, valid_loader, criterion, device, start_time=None):
    model.eval()
    total_loss = 0
    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for drug_feat, prot_feat, labels in valid_loader:
            drug_feat, prot_feat, labels = drug_feat.to(device), prot_feat.to(device), labels.long().to(device)

            outputs = model(drug_feat, prot_feat)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            scores = F.softmax(outputs, 1)
            _, preds = torch.max(scores, 1)
            probs = scores[:, 1]

            y_true.extend(labels.tolist())
            y_pred.extend(preds.tolist())
            y_prob.extend(probs.tolist())

    avg_loss = total_loss / len(valid_loader)
    metrics = calculate_metrics(y_true, y_pred, y_prob)
    duration = time.time() - start_time if start_time else None

    return avg_loss, metrics, duration


def train_model(model, train_loader, valid_loader, optimizer, criterion, device, num_epochs):
    best_val_auc = 0
    best_model = None
    best_metrics = None
    valid_metrics_all = []
    separator = '-' * 130

    early_stopper = EarlyStopping(patience=10, delta=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5,
                                                           threshold=0.0001, threshold_mode='rel', cooldown=0,
                                                           min_lr=1e-6, eps=1e-08, verbose=False)

    print(">>>>>>>>>>>>>>>>>>>>>>>>> Starting training >>>>>>>>>>>>>>>>>>>>>>>>")
    print("%10s %15s %15s %15s %15s %15s %15s %15s %15s" %
          ("Epoch", "TrainLoss", "ValLoss", "ValACC", "ValPre", "ValRec", "ValF1", "ValAUC_ROC", "ValAUC_PR"))
    print(separator)

    for epoch in range(num_epochs):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        valid_loss, metrics, dur = validate(model, valid_loader, criterion, device, start)

        val_acc = metrics['Accuracy']
        val_pre = metrics['Precision']
        val_rec = metrics['Recall']
        val_f1 = metrics['F1']
        val_roc = metrics['AUROC']
        val_pr = metrics['AUPR']

        scheduler.step(valid_loss)

        print("%10s %15.4f %15.4f %15.4f %15.4f %15.4f %15.4f %15.4f %15.4f" %
              (f"{epoch+1}/{num_epochs}", train_loss, valid_loss, val_acc,
               val_pre, val_rec, val_f1, val_roc, val_pr))

        valid_metrics_all.append([train_loss, valid_loss, val_acc, val_pre, val_rec, val_f1, val_roc, val_pr])

        if val_roc > best_val_auc:
            best_val_auc = val_roc
            best_model = copy.deepcopy(model)
            best_metrics = (epoch, val_acc, val_pre, val_rec, val_f1, val_roc, val_pr)
        
        early_stopper.step(val_roc)
        if early_stopper.early_stop:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    print("\nBest validation performance:")
    print("%15s %15s %15s %15s %15s %15s %15s" %
          ("Epoch", "ACC", "Pre", "Rec", "F1", "AUC", "PR"))
    print("%10d %15.4f %15.4f %15.4f %15.4f %15.4f %15.4f" % best_metrics)

    return best_model, valid_metrics_all


def test_model(model, test_loader, criterion, device):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    total_loss = 0

    with torch.no_grad():
        for drug_feat, prot_feat, labels in test_loader:
            drug_feat, prot_feat, labels = drug_feat.to(device), prot_feat.to(device), labels.long().to(device)

            outputs = model(drug_feat, prot_feat)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            scores = F.softmax(outputs, 1)
            _, pred = torch.max(scores, 1)
            prob = scores[:, 1]

            y_true.extend(labels.tolist())
            y_pred.extend(pred.tolist())
            y_prob.extend(prob.tolist())

    metrics = calculate_metrics(y_true, y_pred, y_prob)
    print("\nTest set performance:")
    print("%15s %15s %15s %15s %15s %15s" %
          ("ACC", "Pre", "Rec", "F1", "ROC", "PR"))
    print("%15.4f %15.4f %15.4f %15.4f %15.4f %15.4f" %
          (metrics['Accuracy'], metrics['Precision'], metrics['Recall'],
           metrics['F1'], metrics['AUROC'], metrics['AUPR']))

    return metrics


if __name__ == '__main__':
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    setup_seed(3407)
    num_epochs = 100

    task = 'Kinase'
    task_lower = task.lower()

    model_dir = f'./models/{task}/'
    os.makedirs(model_dir, exist_ok=True)

    log_dir = f'./logs/{task}/'
    os.makedirs(log_dir, exist_ok=True)

    result_dir = f'./results/{task}/'
    os.makedirs(result_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f'random_{task_lower}.log')
    sys.stdout = Logger(log_file)

    seed_results = []

    input_dir = f'/home/qfchen/ProteinDrugInter/datasets/{task}/random_split_results/'
    seeds = [42, 1234, 3407, 5678, 8888]
    for i_seed in seeds:
        print('*' * 25, 'No.', i_seed, '-seed', '*' * 25)
        train_file = input_dir + f'seed_{i_seed}/random_train.csv'
        valid_file = input_dir + f'seed_{i_seed}/random_val.csv'
        test_file = input_dir + f'seed_{i_seed}/random_test.csv'

        train_set = LoadMyData(train_file, f'{task_lower}')
        valid_set = LoadMyData(valid_file, f'{task_lower}')
        test_set = LoadMyData(test_file, f'{task_lower}')
        print(f"Train set: {len(train_set)} and Valid set: {len(valid_set)} and Test set: {len(test_set)}")

        train_dataset = CustomDataset(train_set)
        valid_dataset = CustomDataset(valid_set)
        test_dataset = CustomDataset(test_set)

        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=32, drop_last=True)
        valid_loader = DataLoader(valid_dataset, shuffle=False, batch_size=32, drop_last=False)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=32, drop_last=False)

        criterion = nn.CrossEntropyLoss()
        model = DNN(input_size=1074, hidden_size=512, output_size=2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

        best_model, _ = train_model(model, train_loader, valid_loader, optimizer, criterion, device, num_epochs)
        torch.save(best_model.state_dict(),
               os.path.join(model_dir, f'random_{task_lower}_seed_{i_seed}.pt'))
        
        test_metrics = test_model(best_model, test_loader, criterion, device)

        seed_results.append({
            'Seed': i_seed,
            'Accuracy': test_metrics['Accuracy'],
            'Precision': test_metrics['Precision'],
            'Recall': test_metrics['Recall'],
            'F1': test_metrics['F1'],
            'AUROC': test_metrics['AUROC'],
            'AUPR': test_metrics['AUPR']})

    # Calculate average metrics across all seeds
    results_df = pd.DataFrame(seed_results).round(3)
    avg_acc = results_df['Accuracy'].mean()
    avg_pre = results_df['Precision'].mean()
    avg_rec = results_df['Recall'].mean()
    avg_f1 = results_df['F1'].mean()
    avg_auroc = results_df['AUROC'].mean()
    avg_aupr = results_df['AUPR'].mean()

    # Standard deviations
    std_acc = results_df['Accuracy'].std()
    std_pre = results_df['Precision'].std()
    std_rec = results_df['Recall'].std()
    std_f1 = results_df['F1'].std()
    std_auroc = results_df['AUROC'].std()
    std_aupr = results_df['AUPR'].std()

    # Print formatted results
    print("\n=== Final 10-seed CV Results ===")
    print(results_df)
    print("\n=== Performance Summary ===")
    print(f"Average Accuracy:  {avg_acc:.3f}±{std_acc:.3f}")
    print(f"Average Precision: {avg_pre:.3f}±{std_pre:.3f}")
    print(f"Average Recall:    {avg_rec:.3f}±{std_rec:.3f}")
    print(f"Average F1:        {avg_f1:.3f}±{std_f1:.3f}")
    print(f"Average AUROC:     {avg_auroc:.3f}±{std_auroc:.3f}")
    print(f"Average AUPR:      {avg_aupr:.3f}±{std_aupr:.3f}")

    # 保存汇总结果
    results_df.to_csv(os.path.join(result_dir, f'random_{task_lower}_cv_results.csv'), index=False)
