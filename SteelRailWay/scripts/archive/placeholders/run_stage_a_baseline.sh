#!/usr/bin/env bash
# Stage A baseline：单视角 TRD 跑通（占位脚本，路径请按实际填）
set -euo pipefail
cd "$(dirname "$0")/.."

python train/train_trd_mvtec3d_rgbd.py \
    --config configs/stage_a_rail_single_view.yaml
