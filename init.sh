#!/bin/bash
# ============================================================================
# 初始化脚本 — 配置 Python 环境 + 数据预处理 + GPU 检测
#
# 用法:
#   Docker:  docker exec -it dbc2026 bash /app/init.sh
#   本地:    bash init.sh
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- 环境检测：Docker 内已有 Python，本地则尝试激活 conda ---
if [ -f /.dockerenv ] || [ -n "$DOCKER_ENV" ]; then
    echo "========================================="
    echo "检测到 Docker 环境，使用系统 Python"
    echo "========================================="
else
    echo "========================================="
    echo "检测到本地环境，尝试激活 conda THU-BDC"
    echo "========================================="

    # 初始化 conda
    if [ -f "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" ]; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate THU-BDC 2>/dev/null || {
            echo "警告: 未找到 THU-BDC 环境，使用系统 Python"
        }
    else
        echo "警告: 未找到 conda，使用系统 Python"
    fi
fi

echo "Python: $(which python)"
echo "Python 版本: $(python --version)"

# --- 安装依赖（仅在本地环境执行；Docker 镜像已预装） ---
if ! [ -f /.dockerenv ] && ! [ -n "$DOCKER_ENV" ]; then
    echo ""
    echo "========================================="
    echo "安装项目依赖"
    echo "========================================="

    # PyTorch (CUDA 12.4)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

    # 其他依赖
    pip install -r requirements.txt

    # TA-Lib C 库检测
    python -c "import talib" 2>/dev/null || {
        echo "尝试通过 conda 安装 TA-Lib..."
        conda install -c conda-forge ta-lib -y 2>/dev/null || {
            echo "请手动安装 TA-Lib C 库后重新运行"
            echo "  Ubuntu: sudo apt-get install ta-lib"
            echo "  或: conda install -c conda-forge ta-lib"
            exit 1
        }
    }

    echo "依赖安装完成"
fi

# --- 步骤 1: 数据划分 ---
echo ""
echo "========================================="
echo "步骤 1/2: 划分训练集/测试集"
echo "========================================="
python code/split_train_test.py

# --- 步骤 2: GPU 检测 ---
echo ""
echo "========================================="
echo "步骤 2/2: GPU 环境检测"
echo "========================================="
python code/test-GPU.py

echo ""
echo "========================================="
echo "初始化完成"
echo "========================================="
