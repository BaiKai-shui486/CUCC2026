#!/bin/bash
# ============================================================================
# 测试脚本 — 滚动窗口预测 + 业绩归因报告
#
# 用法:
#   Docker:  docker exec -it dbc2026 bash /app/test.sh
#   本地:    bash test.sh
#
# 可选参数:
#   bash test.sh --mode single    # 仅预测最后一个窗口，输出 result.csv
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- 环境检测 ---
if [ -f /.dockerenv ] || [ -n "$DOCKER_ENV" ]; then
    echo "========================================="
    echo "Docker 环境 — 开始测试"
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

# --- 运行测试（支持传入额外参数） ---
python code/src/test.py "$@"

echo ""
echo "========================================="
echo "测试完成"
echo "========================================="
