#!/usr/bin/env bash
# Stage A 消融：深度归一化、输入尺寸、单/双模态、CF/CA 移除（占位）
set -euo pipefail
cd "$(dirname "$0")/.."

# A2: 深度归一化策略
for norm in zscore minmax log; do
    echo "=== A2 depth_norm=${norm} ==="
    # python train/train_trd_mvtec3d_rgbd.py --config configs/stage_a_rail_single_view.yaml --override data.depth_norm=${norm}
done

# A4: 单模态消融
# python ... --override modality=intensity_only
# python ... --override modality=depth_only

# A5: 移除 CF / CA
# python ... --override model.use_cf=false
# python ... --override model.use_ca=false
