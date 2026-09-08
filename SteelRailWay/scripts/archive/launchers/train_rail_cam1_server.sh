#!/bin/bash
# 钢轨数据集 TRD 训练启动脚本 (Linux 服务器版本)

echo "========================================"
echo "Starting TRD Training on Rail Dataset"
echo "========================================"

# 设置参数
TRAIN_ROOT="/data1/Leaddo_data/20260327"
TEST_ROOT="/home/root123/LF/WYM/SteelRailWay/rail_mvtec_gt_test"
VIEW_ID=1
BATCH_SIZE=16
EPOCHS=100
LR=0.0001
DEVICE="cuda:0"
SAVE_DIR="./outputs/rail"

# 数据采样参数（默认使用 10% 数据，约 2940 张图像）
SAMPLE_RATIO=0.1
# 或者使用固定数量（取消注释下面这行，会覆盖 SAMPLE_RATIO）
# SAMPLE_NUM=3000

# 创建输出目录
mkdir -p "$SAVE_DIR"

# 开始训练
python train/train_trd_rail.py \
    --train_root "$TRAIN_ROOT" \
    --test_root "$TEST_ROOT" \
    --view_id $VIEW_ID \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --device "$DEVICE" \
    --save_dir "$SAVE_DIR" \
    --eval_interval 5 \
    --num_workers 4 \
    --use_patch True \
    --patch_size 900 \
    --patch_stride 850 \
    --depth_norm zscore \
    --train_sample_ratio $SAMPLE_RATIO \
    ${SAMPLE_NUM:+--train_sample_num $SAMPLE_NUM}

echo "========================================"
echo "Training finished!"
echo "========================================"
