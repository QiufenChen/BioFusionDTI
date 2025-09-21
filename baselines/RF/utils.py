#!/usr/bin/env python
# Author  : KerryChen
# File    : utils.py
# Time    : 2025/6/26 11:31
import sys

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