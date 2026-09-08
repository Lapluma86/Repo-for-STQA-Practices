#!/bin/bash
# 串行训练所有 8 个 Cam 的脚本（服务器版本）

echo "========================================"
echo "Serial Training for Multiple Cameras"
echo "========================================"

# 设置公共参数
TRAIN_ROOT="/data1/Leaddo_data/20260327"
TEST_ROOT="/home/root123/LF/WYM/SteelRailWay/rail_mvtec_gt_test"
BATCH_SIZE=64
EPOCHS=100
LR=0.0002
DEVICE="cuda:0"
SAVE_DIR="./outputs/rail"

# 数据采样参数
SAMPLE_RATIO=0.1

# 预加载参数
PRELOAD_WORKERS=32

# Triton/torch.compile 需要 libcuda.so 用于链接阶段
export LIBRARY_PATH="$HOME/lib:$LIBRARY_PATH"

# 数据划分比例（train:val:test = 8:1:1）
TRAIN_VAL_TEST_SPLIT="0.8 0.1 0.1"

# 解析命令行参数
if [ $# -eq 0 ]; then
    # 没有参数，训练所有 8 个 Cam
    CAMS=(1 2 3 4)
    echo "Training all 8 cameras: ${CAMS[@]}"
else
    # 有参数，训练指定的 Cam
    CAMS=("$@")
    echo "Training specified cameras: ${CAMS[@]}"
fi

# 创建输出目录
mkdir -p "$SAVE_DIR"

# 记录开始时间
START_TIME=$(date +%s)

# 串行训练每个 Cam
for VIEW_ID in "${CAMS[@]}"; do
    echo ""
    echo "========================================"
    echo "Training Camera $VIEW_ID"
    echo "========================================"

    CAM_START_TIME=$(date +%s)

    python train/train_trd_rail.py \
        --train_root "$TRAIN_ROOT" \
        --test_root "$TEST_ROOT" \
        --view_id $VIEW_ID \
        --batch_size $BATCH_SIZE \
        --epochs $EPOCHS \
        --lr $LR \
        --device "$DEVICE" \
        --save_dir "$SAVE_DIR" \
        --eval_interval 10 \
        --num_workers 16 \
        --use_patch True \
        --patch_size 900 \
        --patch_stride 850 \
        --depth_norm zscore \
        --train_sample_ratio $SAMPLE_RATIO \
        --train_val_test_split $TRAIN_VAL_TEST_SPLIT \
        --preload \
        --preload_workers $PRELOAD_WORKERS

    CAM_END_TIME=$(date +%s)
    CAM_DURATION=$((CAM_END_TIME - CAM_START_TIME))

    echo ""
    echo "Camera $VIEW_ID training completed in $CAM_DURATION seconds"
    echo "========================================"
done

# 记录结束时间
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

echo ""
echo "========================================"
echo "All training finished!"
echo "Total time: $TOTAL_DURATION seconds"
echo "========================================"
