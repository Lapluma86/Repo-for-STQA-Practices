#!/usr/bin/env bash
# ============================================================
# 全量 HDF5 打包脚本 — 将 train + val 全部压缩为 .h5
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

DATA_ROOT="${ADODAS_DATA_ROOT:-/data1/AdoDas}"
OUT_DIR="${ADODAS_HDF5_DIR:-$DATA_ROOT}"
CONFIG_ARG=""

usage() {
    cat << 'EOF'
用法: ./scripts/pack_all.sh [选项]

选项:
  --config <path>   指定训练 YAML 配置文件，用于设置正确的 SSL 模型标签
                    不指定则使用默认值 (audio: chinese-hubert-base, video: dinov2-base)
                    推荐: --config tasks/a2/default.yaml
  -h, --help        显示此帮助

环境变量:
  ADODAS_DATA_ROOT   数据根目录 (默认: /data1/AdoDas)
  ADODAS_HDF5_DIR    HDF5 输出目录 (默认: 同数据根目录)

示例:
  ./scripts/pack_all.sh
  ./scripts/pack_all.sh --config tasks/a2/default.yaml
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG_ARG="--config $2"; shift 2 ;;
        -h|--help)
            usage ;;
        *)
            echo "未知参数: $1"; usage ;;
    esac
done

echo "============================================================"
echo " ADODAS 2026 — Full Dataset HDF5 Packing"
echo "============================================================"
echo " Data root:  $DATA_ROOT"
echo " Output dir: $OUT_DIR"
[[ -n "$CONFIG_ARG" ]] && echo " Config:     $CONFIG_ARG"
echo "============================================================"

# ---- Train ----
echo ""
echo "[1/2] Packing train set ..."
# shellcheck disable=SC2086
python "$PROJECT_DIR/scripts/pack_features.py" \
    --manifest "$DATA_ROOT/Train/train.csv" \
    --feature-root "$DATA_ROOT" \
    --split train \
    --output "$OUT_DIR/train_packed.h5" \
    --compression gzip \
    $CONFIG_ARG

TRAIN_SIZE=$(du -h "$OUT_DIR/train_packed.h5" | cut -f1)
echo "  Done: $OUT_DIR/train_packed.h5 ($TRAIN_SIZE)"

# ---- Val ----
echo ""
echo "[2/2] Packing validation set ..."
# shellcheck disable=SC2086
python "$PROJECT_DIR/scripts/pack_features.py" \
    --manifest "$DATA_ROOT/Val/val.csv" \
    --feature-root "$DATA_ROOT" \
    --split val \
    --output "$OUT_DIR/val_packed.h5" \
    --compression gzip \
    $CONFIG_ARG

VAL_SIZE=$(du -h "$OUT_DIR/val_packed.h5" | cut -f1)
echo "  Done: $OUT_DIR/val_packed.h5 ($VAL_SIZE)"

# ---- Summary ----
echo ""
echo "============================================================"
echo " Packing complete"
echo "============================================================"
echo " Train: $OUT_DIR/train_packed.h5 ($TRAIN_SIZE)"
echo " Val:   $OUT_DIR/val_packed.h5 ($VAL_SIZE)"
echo ""
echo " Usage: set use_hdf5: true in config YAML"
echo "============================================================"
