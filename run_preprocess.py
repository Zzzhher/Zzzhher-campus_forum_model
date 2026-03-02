#!/usr/bin/env python3
"""
运行数据预处理
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import DataPreprocessor
from config import Config

if __name__ == "__main__":
    dp = DataPreprocessor(Config)
    dp.run()
