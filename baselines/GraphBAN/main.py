# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 17:56:40 2025

@author: 18811
"""

import sys
sys.path.append('/home/qfchen/GraphBAN')
import os
import pandas as pd
import time
from sklearn.model_selection import train_test_split, KFold
import subprocess
import shutil
import warnings
warnings.filterwarnings('ignore')

print("Benchmark of GraphBAN")


if __name__ == '__main__':
    #1. 读入数据
    path = r"/home/qfchen/GraphBAN/files"
    dataset = pd.read_csv(os.path.join(path, 'split_dataset.tsv'), sep='\t')
    dataset = dataset[['SMILES','sequence','label']]
    dataset.columns = ['SMILES','Protein','Y']
    
    random_state = 42
    kf = KFold(n_splits=10, shuffle=True, random_state=random_state)
    code_file = r"/home/qfchen/GraphBAN/inductive_mode"
    
    for i, (tv_index, test_index) in enumerate(kf.split(dataset)):
        
        print(f"Fold {i+1}:")
        s = time.time()
        tv_set, test_set = dataset.iloc[tv_index,:], dataset.iloc[test_index,:]
        train_set, val_set = train_test_split(tv_set, test_size=1/9, random_state=random_state,
                                              shuffle=True, stratify=tv_set.Y)
        
        train_set = train_set.reset_index(drop=True)
        val_set = val_set.reset_index(drop=True)
        test_set = test_set.reset_index(drop=True)
        
        train_set.to_csv(os.path.join(path, f'train_set_{i}.csv'),index=False)
        val_set.to_csv(os.path.join(path, f'val_set_{i}.csv'),index=False)
        test_set.to_csv(os.path.join(path, f'test_set_{i}.csv'),index=False)
        
        # 生成教师模块图的embedding
        code = os.path.join(code_file,'teacher_gae.py')
        train_path = os.path.join(path, f'train_set_{i}.csv')
        teacher_path = os.path.join(path, f'teacher_emd_{i}.parquet')
        cmd = f'python {code} --train_path {train_path} --seed 42 --teacher_path {teacher_path} --epoch 400'
        subprocess.call(cmd, cwd=code_file, shell=True)
        
        # 训练transductive mode
        run_path = os.path.join(code_file,'run_model.py')
        val_path = os.path.join(path, f'val_set_{i}.csv')
        test_path = os.path.join(path, f'test_set_{i}.csv')
        cmd = f'python {run_path} --train_path {train_path} --val_path {val_path} ' +\
              f'--test_path {test_path} --seed 42 ' +\
              f'--mode transductive --teacher_path {teacher_path}'
        subprocess.call(cmd, cwd=code_file, shell=True)
        
        # 删除中间文件，train/val/test_set_i.csv以及teacher_emd_i.parquet
        os.remove(train_path)
        os.remove(val_path)
        os.remove(test_path)
        os.remove(teacher_path)
        
        
       
