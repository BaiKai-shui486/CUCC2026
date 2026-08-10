#!/bin/bash
# ============================================================================
# Docker 镜像构建 & 导出脚本
#
# 用法:
#   bash app/docker/export.sh          # 构建 dbc2026 镜像并导出为 GEMN.tar
#   bash app/docker/export.sh build    # 仅构建镜像
#   bash app/docker/export.sh export   # 仅导出已有镜像
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(dirname "$SCRIPT_DIR")"
IMAGE_NAME="bdc2026:latest"
TAR_FILE="$SCRIPT_DIR/GEMN.tar"

MODE="${1:-all}"

# --- 构建 ---
if [ "$MODE" = "all" ] || [ "$MODE" = "build" ]; then
    echo "========================================="
    echo "构建 Docker 镜像: $IMAGE_NAME"
    echo "构建上下文: $APP_DIR"
    echo "Dockerfile:  $SCRIPT_DIR/Dockerfile"
    echo "========================================="

    docker build \
        -f "$SCRIPT_DIR/Dockerfile" \
        -t "$IMAGE_NAME" \
        "$APP_DIR"

    echo ""
    echo "镜像构建完成: $IMAGE_NAME"
    docker images "$IMAGE_NAME"
fi

# --- 导出 ---
if [ "$MODE" = "all" ] || [ "$MODE" = "export" ]; then
    echo ""
    echo "========================================="
    echo "导出镜像为: $TAR_FILE"
    echo "========================================="

    docker save -o "$TAR_FILE" "$IMAGE_NAME"

    FILE_SIZE=$(du -h "$TAR_FILE" | cut -f1)
    echo ""
    echo "导出完成: $TAR_FILE ($FILE_SIZE)"
    echo ""
    echo "接收方可通过以下命令加载镜像:"
    echo "  docker load -i GEMN.tar"
fi
