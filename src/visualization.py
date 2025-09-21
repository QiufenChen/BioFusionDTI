#!/usr/bin/env python
# Author  : KerryChen
# File    : test1.py.py
# Time    : 2025/2/28 15:38


import copy
import json
import os
import random
import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
from prefetch_generator import BackgroundGenerator
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from matplotlib import cm
from matplotlib import colors as mcolors
from load_data import LoadMyData, CustomDataset
from model import DTIPredictor
import torch.nn.functional as F
import dgl
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, precision_recall_curve, auc)


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

def plot_heat_map(output, suffix):
    plt.figure(figsize=(30, 2))
    sns.heatmap(output.cpu().numpy(), cmap="Reds", cbar=True, yticklabels=False)
    plt.title(f"{suffix} Values")
    plt.xlabel("Feature Index")
    plt.savefig('figures/heat_map_' + suffix + '.png', dpi=600)
    # plt.show()


def plot_attention_heatmap(attention_weights, drug_mask, prot_mask, seed, sample_idx, save_dir='figures'):
    """
    Plot attention weights as separate 1D heatmaps for drug and protein
    
    Args:
        attention_weights: tensor of shape (batch_size, h_out, v_num, q_num)
        drug_mask: tensor indicating valid drug nodes
        prot_mask: tensor indicating valid protein nodes  
        sample_idx: which sample in batch to visualize
        save_dir: directory to save heatmaps
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract attention weights for the specific sample
    att_weights = attention_weights[0]  # (h_out, v_num, q_num)
    
    # Apply masks to focus on valid nodes
    drug_mask_sample = drug_mask[0]  # (v_num,)
    prot_mask_sample = prot_mask[0]  # (q_num,)
    
    # Get valid drug and protein indices
    valid_drug_indices = torch.where(drug_mask_sample == 1)[0]
    valid_prot_indices = torch.where(prot_mask_sample == 1)[0]
    
    # Average attention weights across all heads
    avg_attention = torch.mean(att_weights, dim=0)  # (v_num, q_num)
    
    # Extract only valid nodes for the averaged attention
    avg_weights = avg_attention[valid_drug_indices][:, valid_prot_indices]
    
    # Convert to numpy for plotting
    weights_np = avg_weights.detach().cpu().numpy()
    
    # Create two 1D heatmaps
    # 1. Drug attention: average across protein axis (shape: (1, num_drug_nodes))
    drug_attention = np.mean(weights_np, axis=1, keepdims=True)  # (num_drug_nodes, 1)
    drug_attention = drug_attention.T  # (1, num_drug_nodes)
    
    # 2. Protein attention: average across drug axis (shape: (1, num_protein_nodes))
    protein_attention = np.mean(weights_np, axis=0, keepdims=True)  # (1, num_protein_nodes)
    
    # Normalize both attention vectors
    scaler_drug = MinMaxScaler(feature_range=(0, 1))
    scaler_protein = MinMaxScaler(feature_range=(0, 1))
    
    drug_attention_scaled = scaler_drug.fit_transform(drug_attention.T).T
    protein_attention_scaled = scaler_protein.fit_transform(protein_attention.T).T
    
    print(f"Drug attention shape: {drug_attention_scaled.shape}")
    print(f"Protein attention shape: {protein_attention_scaled.shape}")

    # cmap = cm.get_cmap('Reds')
    # drug_rgb = [cmap(v)[:3] for v in drug_attention_scaled.flatten()]
    # prot_rgb = [cmap(v)[:3] for v in protein_attention_scaled.flatten()]
    #
    drug_df = pd.DataFrame({
        "chain": "A",
        "drug_node_index": valid_drug_indices.detach().cpu().numpy(),
        "attention_norm": drug_attention_scaled.flatten(),
    })
    prot_df = pd.DataFrame({
        "chain": "A",
        "protein_residue_index": valid_prot_indices.detach().cpu().numpy(),
        "attention_norm": protein_attention_scaled.flatten(),
    })

    drug_csv = os.path.join(save_dir, f"{seed}_drug_colors_sample_{sample_idx}.csv")
    prot_csv = os.path.join(save_dir, f"{seed}_protein_colors_sample_{sample_idx}.csv")
    drug_df.to_csv(drug_csv, index=False)
    prot_df.to_csv(prot_csv, index=False)
    print(f"[Saved] {drug_csv}")
    print(f"[Saved] {prot_csv}")
    
    # Create separate figure for drug attention
    plt.figure(figsize=(12, 4))
    im1 = plt.imshow(drug_attention_scaled, cmap='Reds', aspect='auto')
    plt.title(f'Drug Attention Weights (Sample {sample_idx}) - Averaged across Protein Axis')
    plt.xlabel('Drug Atoms')
    plt.ylabel('Attention')
    plt.yticks([])  # Remove y-axis ticks for cleaner look
    
    # Add colorbar for drug attention
    cbar1 = plt.colorbar(im1)
    cbar1.set_label('Normalized Attention Weight')
    
    # Add text annotation for drug attention
    plt.text(0.02, 0.98, f'Shape: {drug_attention_scaled.shape}\nAveraged across {weights_np.shape[1]} protein nodes', 
             transform=plt.gca().transAxes, 
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{seed}_drug_attention_sample_{sample_idx}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_dir}/{seed}_drug_attention_sample_{sample_idx}.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    
    # Create separate figure for protein attention
    plt.figure(figsize=(12, 4))
    im2 = plt.imshow(protein_attention_scaled, cmap='Reds', aspect='auto')
    plt.title(f'Protein Attention Weights (Sample {sample_idx}) - Averaged across Drug Axis')
    plt.xlabel('Protein Residue')
    plt.ylabel('Attention')
    plt.yticks([])  # Remove y-axis ticks for cleaner look
    
    # Add colorbar for protein attention
    cbar2 = plt.colorbar(im2)
    cbar2.set_label('Normalized Attention Weight')
    
    # Add text annotation for protein attention
    plt.text(0.02, 0.98, f'Shape: {protein_attention_scaled.shape}\nAveraged across {weights_np.shape[0]} drug nodes', 
             transform=plt.gca().transAxes, 
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{seed}_protein_attention_sample_{sample_idx}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_dir}/{seed}_protein_attention_sample_{sample_idx}.pdf', format='pdf', bbox_inches='tight')

    plt.close()
    
    return drug_attention_scaled, protein_attention_scaled


def test_labeled(model, loader, criterion, device, task, seed, case):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    test_loss = 0.0
    all_attention_weights = []
    all_drug_masks = []
    all_prot_masks = []
    results = []

    save_dir = f'./Visualization/{task}/drug/{case}/'
    os.makedirs(save_dir, exist_ok=True)

    for batch_idx, (v_dg, v_pg, v_dm, v_pm, v_df, v_pf, labels) in tqdm(enumerate(BackgroundGenerator(loader)), total=len(loader)):
        v_dg = v_dg.to(device)
        v_pg = v_pg.to(device)
        v_dm = v_dm.to(device)
        v_pm = v_pm.to(device)
        v_df = v_df.to(device)
        v_pf = v_pf.to(device)
        labels = labels.long().to(device)

        att, outputs = model(v_dg, v_pg, v_dm, v_pm, v_df, v_pf)
        
        # Store attention weights and masks for visualization
        all_attention_weights.append(att.detach().cpu())
        all_drug_masks.append(v_dm.detach().cpu())
        all_prot_masks.append(v_pm.detach().cpu())
        
        print(f"Sample {batch_idx}:")
        print(f"  Attention shape: {att.shape}")
        print(f"  Drug mask shape: {v_dm.shape}")
        print(f"  Protein mask shape: {v_pm.shape}")
        print(f"  Valid drug nodes: {torch.sum(v_dm).item()}")
        print(f"  Valid protein nodes: {torch.sum(v_pm).item()}")
        

        drug_att, protein_att = plot_attention_heatmap(att, v_dm, v_pm, seed, sample_idx=batch_idx, save_dir=save_dir)
        
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        scores = F.softmax(outputs, dim=1)
        _, pred = torch.max(scores, dim=1)
        prob = scores[:, 1]

        # ⭐ 保存结果
        results.append({
            "Seed": seed,
            "Sample": batch_idx,
            "True": labels.item(),
            "Pred": pred.item(),
            "Prob": prob.item()
        })

        print(f"Sample {batch_idx}: Prob={prob.item():.3f}, Pred={pred.item()}, True={labels.item()}")

    # 转成 DataFrame
    df_results = pd.DataFrame(results)
    save_path = save_dir + f"/predictions_seed_{seed}.csv"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_results.to_csv(save_path, index=False)
    print(f"[Saved predictions] {save_path}")

    # Create summary visualization
    if all_attention_weights:
        # Concatenate all attention weights
        combined_attention = torch.cat(all_attention_weights, dim=0)
        combined_drug_masks = torch.cat(all_drug_masks, dim=0)
        combined_prot_masks = torch.cat(all_prot_masks, dim=0)


        print(f"\nVisualization Summary:")
        print(f"  Total samples processed: {len(all_attention_weights)}")
        print(f"  Attention weights shape: {combined_attention.shape}")
        print(f"  Heatmaps saved to: figures/")

    # metrics = calculate_metrics(y_true, y_pred, y_prob)
    # return y_true, y_pred, y_prob, test_loss / len(loader), metrics


def run_seeds_test(model_class, test_loader, criterion, model_dir, task, device, seeds, case):
    all_results = []
    for seed in seeds:
        print(f"\n{'*'*25} Seed {seed} {'*'*25}")
        model = model_class().to(device)
        model_path = f"{model_dir}/drug_{task}_seed_{seed}.pt"
        model.load_state_dict(torch.load(model_path, map_location=device))

        # _, _, _, loss, metrics = test_labeled(model, test_loader, criterion, device)
        test_labeled(model, test_loader, criterion, device, task, seed, case)
        # print(f"[Test] Loss: {loss:.4f} |", " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items()))

        # all_results.append({**{'Seed': seed}, **metrics})

    # return pd.DataFrame(all_results).round(3)


def print_summary(df: pd.DataFrame):
    print("\n=== Performance Summary ===")
    for metric in df.columns[1:]:
        avg = df[metric].mean()
        std = df[metric].std()
        print(f"{metric:12}: {avg:.3f} ± {std:.3f}")


if __name__ == '__main__':
    setup_seed(3407)
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    task = 'SNAP'
    case = 'EGFR'
    seeds = [42, 1234, 3407, 5678, 8888]
    # seeds = [8888]
    # Prepare data
    test_file = f'/home/qfchen/ProteinDrugInter/datasets/ExternalDatasets/{case}.xlsx'

    df_test = pd.read_excel(test_file, engine="openpyxl")
    df_test['Index'] = df_test.index
    print("DF_TEST-------: ", len(df_test))
    test_data = LoadMyData(test_file, labeled=True)
    test_loader = DataLoader(CustomDataset(test_data), batch_size=1, shuffle=False,
                             drop_last=False, collate_fn=collate_func_train)

    # Run evaluation
    criterion = nn.CrossEntropyLoss()
    model_dir = f'./models/{task}'
    run_seeds_test(DTIPredictor, test_loader, criterion, model_dir, task.lower(), device, seeds, case)

