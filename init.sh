#!/bin/bash
# 初始化脚本 - 安装依赖到 conda THU-BDC 环境
# 用法: bash init.sh

set -e

# 激活 conda 环境
conda activate THU-BDC

echo "========================================="
echo "安装项目依赖"
echo "Python: $(which python)"
echo "========================================="

# 安装 PyTorch (CUDA 12.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 安装其他依赖
pip install pandas numpy scikit-learn tqdm tensorboard tensorboardX joblib

# 安装 TA-Lib (先尝试 conda，再尝试 pip)
conda install -c conda-forge ta-lib -y 2>/dev/null || pip install TA-Lib

echo "========================================="
echo "依赖安装完成"
echo "========================================="
