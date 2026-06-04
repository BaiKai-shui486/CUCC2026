#!/bin/bash
# 训练脚本 - 使用 conda THU-BDC 环境
# 用法: bash train.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 激活 conda 环境
conda activate THU-BDC

echo "========================================="
echo "开始训练 StockTransformer 排序模型"
echo "Python: $(which python)"
echo "工作目录: $(pwd)"
echo "========================================="

python code/src/train.py

echo "========================================="
echo "训练完成"
echo "========================================="
