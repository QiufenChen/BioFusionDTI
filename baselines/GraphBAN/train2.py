# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 17:56:40 2025

@author: 18811
"""

import sys
# sys.path.append('/home/qfchen/ProteinDrugInter/CompareModel/GraphBAN')
import os
import pandas as pd
import time
import subprocess
import shutil
import warnings
warnings.filterwarnings('ignore')

print("Benchmark of GraphBAN")


if __name__ == '__main__':
    #1. 读入数据
    path = r"./files/"
    os.makedirs(path, exist_ok=True)
    # dataset = pd.read_csv(os.path.join(path, 'split_dataset.tsv'), sep='\t')
    # dataset = dataset[['SMILES','sequence','label']]
    # dataset.columns = ['SMILES','Protein','Y']
    
    random_state = 42
    seeds = [42, 1234, 3407, 5678, 8888]
    # seeds = [5678, 8888]
    code_file = r"./inductive_mode/"

    task = 'Kinase'
    task_lower = task.lower()

    dataFolder = f'/home/qfchen/ProteinDrugInter/datasets/{task}/drug_split_results/'
    for i_seed in seeds:
        s = time.time()
        print('*' * 25, 'No.', i_seed, '-seed', '*' * 25)

        train_path = dataFolder + f'seed_{i_seed}/drug_train.csv' 
        val_path = dataFolder + f'seed_{i_seed}/drug_val.csv' 
        test_path = dataFolder + f'seed_{i_seed}/drug_test.csv'

        
        # 生成教师模块图的embedding
        code = os.path.join(code_file,'teacher_gae.py')
        teacher_path = os.path.join(path, f'teacher_emd_drug_{task}_{i_seed}.parquet')
        cmd = f'/home/qfchen/software/anaconda3/envs/MyEnv/bin/python {code} --train_path {train_path} --seed 42 --teacher_path {teacher_path} --epoch 10'
        subprocess.call(cmd, shell=True)
        
        # 训练transductive mode
        run_path = os.path.join(code_file,'run_model.py')
        save_dir = f'./results/{task}/drug_split/{i_seed}/'
        os.makedirs(save_dir, exist_ok=True)

        cmd = f'/home/qfchen/software/anaconda3/envs/MyEnv/bin/python {run_path} --train_path {train_path} --val_path {val_path} ' +\
              f'--test_path {test_path} --seed 42 ' +\
              f'--mode transductive --teacher_path {teacher_path} ' +\
              f'--save_dir {save_dir}'
        subprocess.call(cmd, shell=True)
        
        # # 删除中间文件，train/val/test_set_i.csv以及teacher_emd_i.parquet
        # os.remove(train_path)
        # os.remove(val_path)
        # os.remove(test_path)
        # os.remove(teacher_path)
        
        
       
