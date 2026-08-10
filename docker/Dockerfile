# ============================================================================
# THU-BDC2026 Docker 镜像 — bdc2026
# 基于 PyTorch 官方 CUDA 镜像，预装所有 Python 依赖
# 将 code、model、config、脚本打包进镜像（data/output/temp 由 compose 挂载）
# ============================================================================

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

LABEL maintainer="THU-BDC2026"
LABEL description="CausalGRUStockScorer ranking model — training & inference"

# --- 安装 TA-Lib C 库 ---
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    wget \
    tar \
    && rm -rf /var/lib/apt/lists/* \
    && wget -q http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib \
    && ./configure --prefix=/usr \
    && make -j1 \
    && make install \
    && cd .. \
    && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz \
    && ldconfig

# --- 设置工作目录 ---
WORKDIR /app

# --- 安装 Python 依赖 ---
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# --- 复制项目文件（data/output/temp 由 docker-compose 挂载） ---
COPY code/          /app/code/
COPY config/        /app/config/
COPY model/         /app/model/
COPY init.sh        /app/init.sh
COPY train.sh       /app/train.sh
COPY test.sh        /app/test.sh
COPY readme.md      /app/readme.md
COPY requirements.txt /app/requirements.txt

# --- 环境变量 ---
ENV LD_LIBRARY_PATH="/usr/lib:/usr/local/lib:${LD_LIBRARY_PATH}"
ENV PYTHONUNBUFFERED=1

# 默认保持容器运行
CMD ["sleep", "infinity"]
