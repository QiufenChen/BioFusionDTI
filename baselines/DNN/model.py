#!/usr/bin/env python
# Author  : KerryChen
# File    : model.py
# Time    : 2025/7/2 9:12
import torch
import torch.nn as nn
import torch.nn.functional as F


class DNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)   # 第一个全连接层
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # 第二个全连接层
        self.fc3 = nn.Linear(hidden_size, output_size)  # 第三个全连接层

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        x = F.dropout(F.relu(self.fc1(x)), 0.1)
        x = F.dropout(F.relu(self.fc2(x)), 0.1)
        x = self.fc3(x)
        return x
