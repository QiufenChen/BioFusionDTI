#!/usr/bin/env python
# Author  : KerryChen
# File    : utils.py
# Time    : 2025/2/28 15:57

import os
import pandas as pd
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_curve, roc_auc_score,
                             precision_recall_curve, auc)
from sklearn.model_selection import KFold, train_test_split
from typing import List, Tuple


def cold_split_10fold(df: pd.DataFrame, entity_col: str, prefix: str, output_dir: str = None, random_state: int = 42) -> \
List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Perform 10-fold cold split (train:val:test) based on unique entities.
    Each fold has 8:1:1 ratio (train:val:test).

    Args:
        df (pd.DataFrame): Input dataframe
        entity_col (str): Column name for cold splitting ('DrugID' or 'TargetID')
        prefix (str): Filename prefix for saving results
        output_dir (str): Output directory to save split CSVs (optional)
        random_state (int): Random seed

    Returns:
        List of tuples: [(train_df, val_df, test_df) for each fold]
    """
    if entity_col not in df.columns:
        raise ValueError(f"Column '{entity_col}' not found in the DataFrame.")

    entities = df[entity_col].unique()
    kf = KFold(n_splits=10, shuffle=True, random_state=random_state)

    folds = []
    for fold_idx, (train_val_idx, test_idx) in enumerate(kf.split(entities)):
        # Split train + val (80%) vs test (20%)
        train_val_entities = entities[train_val_idx]
        test_entities = entities[test_idx]

        # Further split train (80%) and val (20%) from train_val_entities
        train_entities, val_entities = train_test_split(
            train_val_entities, test_size=(1/9), random_state=random_state  # 0.25 * 80% = 20% val
        )

        # Filter dataframes
        train_df = df[df[entity_col].isin(train_entities)]
        val_df = df[df[entity_col].isin(val_entities)]
        test_df = df[df[entity_col].isin(test_entities)]

        print(f"\n=== Fold {fold_idx + 1} (Cold 10-Fold Split by {entity_col}) ===")
        print(f"Train: {len(train_df)} ({len(train_df) / len(df):.1%}) | Unique {entity_col}: {len(train_entities)}")
        print(f"Validation: {len(val_df)} ({len(val_df) / len(df):.1%}) | Unique {entity_col}: {len(val_entities)}")
        print(f"Test: {len(test_df)} ({len(test_df) / len(df):.1%}) | Unique {entity_col}: {len(test_entities)}")

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            fold_dir = os.path.join(output_dir, f"fold_{fold_idx + 1}")
            os.makedirs(fold_dir, exist_ok=True)

            train_df.to_csv(os.path.join(fold_dir, f'{prefix}_train.csv'), index=False)
            val_df.to_csv(os.path.join(fold_dir, f'{prefix}_val.csv'), index=False)
            test_df.to_csv(os.path.join(fold_dir, f'{prefix}_test.csv'), index=False)

            print(f"Saved to: {fold_dir}")

        folds.append((train_df, val_df, test_df))

    return folds


def random_split_10fold(df: pd.DataFrame, prefix: str, output_dir: str = None, random_state: int = 42) -> List[
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Perform 10-fold random split (train:val:test).
    Each fold has 8:1:1 ratio (train:val:test).

    Args:
        df (pd.DataFrame): Input dataframe
        prefix (str): Filename prefix for saving results
        output_dir (str): Output directory to save split CSVs (optional)
        random_state (int): Random seed

    Returns:
        List of tuples: [(train_df, val_df, test_df) for each fold]
    """
    kf = KFold(n_splits=10, shuffle=True, random_state=random_state)
    folds = []

    for fold_idx, (train_val_idx, test_idx) in enumerate(kf.split(df)):
        # Split train + val (80%) vs test (20%)
        train_val_df = df.iloc[train_val_idx]
        test_df = df.iloc[test_idx]

        # Further split train (80%) and val (20%) from train_val_df
        train_df, val_df = train_test_split(
            train_val_df, test_size=(1/9), random_state=random_state  # 0.25 * 80% = 20% val
        )

        print(f"\n=== Fold {fold_idx + 1} (Random 10-Fold Split) ===")
        print(f"Train: {len(train_df)} ({len(train_df) / len(df):.1%})")
        print(f"Validation: {len(val_df)} ({len(val_df) / len(df):.1%})")
        print(f"Test: {len(test_df)} ({len(test_df) / len(df):.1%})")

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            fold_dir = os.path.join(output_dir, f"fold_{fold_idx + 1}")
            os.makedirs(fold_dir, exist_ok=True)
            train_df.to_csv(os.path.join(fold_dir, f'{prefix}_train.csv'), index=False)
            val_df.to_csv(os.path.join(fold_dir, f'{prefix}_val.csv'), index=False)
            test_df.to_csv(os.path.join(fold_dir, f'{prefix}_test.csv'), index=False)
            print(f"Saved to: {fold_dir}")
        folds.append((train_df, val_df, test_df))

    return folds



class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        # timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")
        # message_with_time = timestamp + message
        self.terminal.write(message)  # 控制台输出
        self.log.write(message)       # 写入文件
        self.flush()                            # 实时保存

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def log_config(self, config_dict):
        """记录配置参数"""
        self.write("\n=== Configuration ===\n")
        for key, value in config_dict.items():
            self.write(f"{key}: {value}" + '\n')
        self.write("======================\n\n")

    def close(self):
        self.flush()
        self.log.close()


class EarlyStopping:
    def __init__(self, patience=10, delta=1e-4):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def binary_cross_entropy(pred_output, labels):
    loss_fct = torch.nn.BCELoss()
    m = nn.Sigmoid()
    n = torch.squeeze(m(pred_output), 1)
    loss = loss_fct(n, labels)
    return n, loss


def cross_entropy_logits(linear_output, label, weights=None):
    class_output = F.log_softmax(linear_output, dim=1)
    n = F.softmax(linear_output, dim=1)[:, 1]
    max_class = class_output.max(1)
    y_hat = max_class[1]  # get the index of the max log-probability
    if weights is None:
        loss = nn.NLLLoss()(class_output, label.type_as(y_hat).view(label.size(0)))
    else:
        losses = nn.NLLLoss(reduction="none")(class_output, label.type_as(y_hat).view(label.size(0)))
        loss = torch.sum(weights * losses) / torch.sum(weights)
    return n, loss


def entropy_logits(linear_output):
    p = F.softmax(linear_output, dim=1)
    loss_ent = -torch.sum(p * (torch.log(p + 1e-5)), dim=1)
    return loss_ent


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

    # Area Under the Receiver Operation Characteristics curve (AUROC)
    auroc = roc_auc_score(y_true, y_prob)

    # Area Under the Precision-Recall Curve (AUPRC)
    pre, rec, _ = precision_recall_curve(y_true, y_prob)
    aupr = auc(rec, pre)  # 注意顺序：recall 是 x 轴，precision 是 y 轴

    return {'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'AUROC': auroc,
            'AUPR': aupr}



def plot_loss_curve(df, save_dir='./', figsize=(7, 5), font='Arial', dpi=600):
    """
    Plot training and validation loss curves.

    Parameters:
        df (DataFrame): DataFrame containing 'TrainLoss' and 'ValidLoss' columns.
        save_path (str, optional): Path where the image will be saved.
        figsize: Size of the figures.
        font: Font family for the plots.
        dpi: DPI for saving the figures.
    """

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Set font and figure size
    plt.rcParams['font.sans-serif'] = font

    plt.figure(figsize=figsize)
    plt.plot(df['Epochs'], df['TrainLoss'], label='Training Loss', color='blue', linewidth=2)
    plt.plot(df['Epochs'], df['ValidLoss'], label='Validation Loss', color='red', linewidth=2)

    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('Loss Over Epoches', fontsize=15, weight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_dir}/loss_curve.png', bbox_inches='tight', dpi=dpi)
    plt.savefig(f'{save_dir}/loss_curve.pdf', bbox_inches='tight', dpi=dpi)
    plt.clf()


def plot_metrics_curve(df, epochs, save_dir='./', figsize=(7, 5), font='Arial', dpi=600):
    """
    Plot the training metrics curves over epochs.

    Parameters:
        df (DataFrame): DataFrame containing metric values with columns 'Accuracy',
                        'Precision', 'Recall', 'F1', 'AUROC', and 'AUPR'.
        epochs (int): Total number of training epochs.
        figsize: Size of the figures.
        font: Font family for the plots.
        dpi: DPI for saving the figures.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Set font and figure size
    plt.rcParams['font.sans-serif'] = font

    plt.figure(figsize=figsize)

    # Define colors for each metric
    colors = {
        'Accuracy': '#FF5733',
        'Precision': '#33B5FF',
        'Recall': '#4CAF50',
        'F1': '#8E44AD',
        'AUROC': '#2C3E50',
        'AUPR': '#2C3E50'
    }

    for metric, color in colors.items():
        plt.plot(range(1, epochs + 1), df[metric], label=metric, color=color, linewidth=2)

    # Set labels and title
    plt.xlabel('Epoch', fontsize=15, weight='bold', labelpad=5)
    plt.ylabel('Metrics', fontsize=15, weight='bold', labelpad=5)
    plt.title('Metrics Over Epoches', fontsize=15, weight='bold')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_dir}/all_metrics.png', bbox_inches='tight', dpi=dpi)
    plt.savefig(f'{save_dir}/all_metrics.pdf', bbox_inches='tight', dpi=dpi)
    plt.clf()


def plot_roc_curve(y_true, y_score, save_dir='./', figsize=(4, 4), font='Arial', dpi=600):
    """
    Plots the ROC curve and FPR/TPR vs. Threshold curves.

    Parameters:
    - y_true: True binary labels.
    - y_score: Target scores (probability estimates of the positive class).
    - save_dir: Directory to save the plots.
    - figsize: Size of the figures.
    - font: Font family for the plots.
    - dpi: DPI for saving the figures.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Compute ROC curve and AUC
    fpr, tpr, thre = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # Set font and figure size
    plt.rcParams['font.sans-serif'] = font

    # Plot ROC curve
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, label=f'AUROC = {roc_auc:.2f}', linewidth=3)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
    plt.xlabel('False Positive Rate', fontsize=16, labelpad=5)
    plt.ylabel('True Positive Rate', fontsize=16, labelpad=5)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_dir}/roc_curve.png', bbox_inches='tight', dpi=dpi)
    plt.savefig(f'{save_dir}/roc_curve.pdf', bbox_inches='tight', dpi=dpi)
    plt.clf()

    # Plot FPR and TPR vs. Threshold
    plt.figure(figsize=figsize)
    plt.plot(thre, fpr, label='FPR', linewidth=3)
    plt.plot(thre, tpr, label='TPR', linewidth=3)
    plt.axvline(x=0.5, ls='--', color='black', linewidth=2)
    plt.xlabel('Threshold', fontsize=16, labelpad=5)
    plt.ylabel('Rate', fontsize=16, labelpad=5)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/fpr_tpr.png', bbox_inches='tight', dpi=dpi)
    plt.savefig(f'{save_dir}/fpr_tpr.pdf', bbox_inches='tight', dpi=dpi)
    plt.clf()


def plot_confusion_matrix(y_true, y_pred, save_dir='./', labels=None, figsize=(4, 4), font='Arial', dpi=600):
    """
    Plot the confusion matrix.
     Parameters:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.
        labels (list, optional): List of label names. Defaults to None.
        save_dir (str, optional): Directory where the plot will be saved. Defaults to './'.
        figsize (tuple, optional): Size of the figure in inches. Defaults to (4, 4).
        font (str, optional): Font style for text. Defaults to 'Arial'.
        dpi (int, optional): Dots per inch for saving the image. Defaults to 600. Parameters:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.
        labels (list, optional): List of label names. Defaults to None.
        save_dir (str, optional): Directory where the plot will be saved. Defaults to './'.
        figsize (tuple, optional): Size of the figure in inches. Defaults to (4, 4).
        font (str, optional): Font style for text. Defaults to 'Arial'.
        dpi (int, optional): Dots per inch for saving the image. Defaults to 600.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Set font and figure size
    plt.rcParams['font.sans-serif'] = font

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    plt.figure(figsize=figsize)
    plt.rcParams['font.sans-serif'] = 'Arial'
    disp.plot(cmap='Blues')
    plt.xlabel('Predicted labels', fontsize=16, labelpad=5)
    plt.ylabel('True labels', fontsize=16, labelpad=5)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title("Confusion Matrix", fontsize=16, weight='bold')
    plt.savefig(f'{save_dir}/confusion_matrix.png', bbox_inches='tight', dpi=dpi)
    plt.savefig(f'{save_dir}/confusion_matrix.pdf', bbox_inches='tight', dpi=dpi)
    plt.clf()

