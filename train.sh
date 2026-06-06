#!/bin/bash
# ============================================================================
# 训练脚本 — 运行 StockTransformer 排序模型训练
#
# 用法:
#   Docker:  docker exec -it dbc2026 bash /app/train.sh
#   本地:    bash train.sh
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- 环境检测 ---
if [ -f /.dockerenv ] || [ -n "$DOCKER_ENV" ]; then
    echo "========================================="
    echo "Docker 环境 — 开始训练"
    echo "========================================="
else
    echo "========================================="
    echo "本地环境 — 激活 conda THU-BDC"
    echo "========================================="

    if [ -f "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" ]; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate THU-BDC 2>/dev/null || true
    fi
fi

echo "Python: $(which python)"
echo "工作目录: $(pwd)"
echo "========================================="

# --- 运行训练 ---
python code/src/train.py

echo ""
echo "========================================="
echo "训练完成"
echo "========================================="
