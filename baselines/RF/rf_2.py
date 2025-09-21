#!/usr/bin/env python
# Author  : KerryChen
# File    : rf_2.py
# Time    : 2025/7/2 21:29

import os
import pickle
import sys
import time

from sklearn.ensemble import RandomForestClassifier

from drug_feature import get_maccs_fingerprint
from prot_feature import get_physic_chemical_propertity
import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    precision_recall_curve, auc, matthews_corrcoef
from utils import Logger

### Data preparation
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

    with open(f'./data/{task}_drug_features.pkl', 'rb') as f:
        drug_dict = pickle.load(f)
        # print(drug_dict)

    with open(f'./data/{task}_protein_features.pkl', 'rb') as f:
        prot_dict = pickle.load(f)
        # print(prot_dict)
    features = []
    for _, row in df.iterrows():
        drug_id = row['DRUG_ID']
        prot_id = row['UNIPROT_ID']
        label = row['Label']

        if drug_id not in drug_dict or prot_id not in prot_dict:
            # print(drug_id, prot_id, label, flush=True)
            continue

        drug_feat = drug_dict[drug_id]
        prot_feat = prot_dict[prot_id]
        combined = np.concatenate([drug_feat, prot_feat, [label]])
        features.append(combined)

    features = np.array(features)
    print(f"Loaded {len(features)} valid pairs (time: {time.time() - t0:.2f}s)")

    return features


if __name__ == '__main__':
    prefix = 'drug'  # ← 可改为 drug / prot / random
    task = 'Kinase'
    task_lower = task.lower()

    log_dir = './logs/'
    os.makedirs(log_dir, exist_ok=True)

    model_dir = './models/'
    os.makedirs(model_dir, exist_ok=True)

    result_dir = './results/'
    os.makedirs(result_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f'{prefix}_rf_{task_lower}.log')
    sys.stdout = Logger(log_file)

    results = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "mcc": [],
        "roc_auc": [],
        "pr_auc": [],
    }

    input_dir = f'/home/qfchen/ProteinDrugInter/datasets/{task}/{prefix}_split_results/'
    seeds = [42, 1234, 3407, 5678, 8888]

    for i in seeds:
        print(f"\n=== Seed {i} ===")
        train_file = os.path.join(input_dir, f'seed_{i}/{prefix}_train.csv')
        valid_file = os.path.join(input_dir, f'seed_{i}/{prefix}_val.csv')
        test_file = os.path.join(input_dir, f'seed_{i}/{prefix}_test.csv')

        # Load and combine train + validation sets
        train_set = LoadMyData(train_file, task_lower)
        valid_set = LoadMyData(valid_file, task_lower)
        test_set = LoadMyData(test_file, task_lower)

        train_set = np.concatenate([train_set, valid_set])
        X_train, y_train = train_set[:, :-1], train_set[:, -1]
        X_test, y_test = test_set[:, :-1], test_set[:, -1]

        # Train Random Forest
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_test)
        y_prob = rf.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        pre = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        roc = roc_auc_score(y_test, y_prob)
        prc, recs, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(recs, prc)

        results["accuracy"].append(acc)
        results["precision"].append(pre)
        results["recall"].append(rec)
        results["f1"].append(f1)
        results["mcc"].append(mcc)
        results["roc_auc"].append(roc)
        results["pr_auc"].append(pr_auc)

        print(f"Accuracy: {acc:.3f}")
        print(f"Precision: {pre:.3f}")
        print(f"Recall: {rec:.3f}")
        print(f"F1-score: {f1:.3f}")
        print(f"MCC: {mcc:.3f}")
        print(f"ROC-AUC: {roc:.3f}")
        print(f"PR-AUC: {pr_auc:.3f}")

    print("\n=== Final 5-Seed Results ===")
    results_df = pd.DataFrame(results).round(3)
    print(results_df)

    for metric in results:
        mean_score = np.mean(results[metric])
        std_score = np.std(results[metric])
        print(f"{metric}: {mean_score:.3f}±{std_score:.3f}")

    results_df.to_csv(os.path.join(result_dir, f'{prefix}_rf_{task_lower}_results.csv'), index=False)


# if __name__ == '__main__':
#     # Store evaluation metrics for each fold
#     log_dir = './logs/'
#     os.makedirs(log_dir, exist_ok=True)

#     model_dir = './models/'
#     os.makedirs(model_dir, exist_ok=True)

#     result_dir = './results/'
#     os.makedirs(result_dir, exist_ok=True)

#     task = 'SNAP'
#     task_lower = task.lower()

#     log_file = os.path.join(log_dir, f'drug_svm_cv_{task_lower}.log')
#     sys.stdout = Logger(log_file)

#     results = {
#         "accuracy": [],
#         "precision": [],
#         "recall": [],
#         "f1": [],
#         "roc_auc": [],
#         "pr_auc": []}

#     input_dir = '/home/qfchen/ProteinDrugInter/datasets/SNAP/drug_split_results/'
#     for i in range(1, 11):
#         print(f"\n=== Fold {i} ===")
#         train_file = input_dir + f'fold_{i}/drug_split_train.csv'
#         valid_file = input_dir + f'fold_{i}/drug_split_val.csv'
#         test_file = input_dir + f'fold_{i}/drug_split_test.csv'

#         # Load and combine train + validation sets
#         train_set = LoadMyData(train_file, 'snap')
#         valid_set = LoadMyData(valid_file, 'snap')
#         test_set = LoadMyData(test_file, 'snap')

#         train_set = np.concatenate([train_set, valid_set])

#         X_train, y_train = train_set[:, :-1], train_set[:, -1]
#         X_test, y_test = test_set[:, :-1], test_set[:, -1]

#         # Initialize and train model
#         svm = SVC(
#             C=1.0,  # 正则化参数（默认1.0）
#             kernel='linear',  # 核函数：'rbf'（高斯核）、'linear'、'poly'等
#             gamma='scale',  # 核系数：'scale'（默认，1/(n_features * X.var())）或 'auto'
#             probability=True,  # 启用概率估计（需为True才能用predict_proba）
#             random_state=42  # 固定随机种子
#         )
#         svm.fit(X_train, y_train)

#         # Predictions
#         y_pred = svm.predict(X_test)
#         y_prob = svm.predict_proba(X_test)[:, 1]  # Probability for ROC-AUC

#         # Calculate metrics
#         acc = accuracy_score(y_test, y_pred)
#         pre = precision_score(y_test, y_pred)
#         rec = recall_score(y_test, y_pred)
#         f1 = f1_score(y_test, y_pred)
#         roc = roc_auc_score(y_test, y_prob)
#         prc, recs, _ = precision_recall_curve(y_test, y_prob)
#         pr_auc = auc(recs, prc)

#         results["accuracy"].append(acc)
#         results["precision"].append(pre)
#         results["recall"].append(rec)
#         results["f1"].append(f1)
#         results["roc_auc"].append(roc)
#         results["pr_auc"].append(pr_auc)

#         print(f"Accuracy: {acc:.4f}")
#         print(f"Precision: {pre:.4f}")
#         print(f"Recall: {rec:.4f}")
#         print(f"F1-score: {f1:.4f}")
#         print(f"ROC-AUC: {roc:.4f}")
#         print(f"PR-AUC: {pr_auc:.4f}")

#     print("\n=== Final 10-Fold CV Results ===")
#     results_df = pd.DataFrame(results)
#     print(results_df)

#     for metric in results:
#         mean_score = np.mean(results[metric])
#         std_score = np.std(results[metric])
#         print(f"{metric}: {mean_score:.4f} ± {std_score:.4f}")

#     results_df.to_csv(os.path.join(result_dir, f'drug_svm_cv_{task_lower}_results.csv'), index=False)
