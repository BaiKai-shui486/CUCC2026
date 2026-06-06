# ============================================================================
# THU-BDC2026 Docker 镜像 — dbc2026
# 基于 PyTorch 官方 CUDA 镜像，预装所有 Python 依赖
# ============================================================================

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

LABEL maintainer="THU-BDC2026"
LABEL description="StockTransformer ranking model — training & inference"

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

# --- 环境变量 ---
ENV LD_LIBRARY_PATH="/usr/lib:/usr/local/lib:${LD_LIBRARY_PATH}"
ENV PYTHONUNBUFFERED=1

# 默认保持容器运行
CMD ["sleep", "infinity"]
