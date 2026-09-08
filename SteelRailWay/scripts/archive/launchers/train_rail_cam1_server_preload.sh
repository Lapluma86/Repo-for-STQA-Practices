#!/bin/bash
# 钢轨数据集 TRD 训练启动脚本 (Linux 服务器版本 - 启用预加载)

echo "========================================"
echo "Starting TRD Training with Preloading"
echo "========================================"

# 设置参数
TRAIN_ROOT="/data1/Leaddo_data/20260327"
TEST_ROOT="/home/root123/LF/WYM/SteelRailWay/rail_mvtec_gt_test"
VIEW_ID=1
BATCH_SIZE=128        # 增大 batch size（显存充足）
EPOCHS=100
LR=0.0002             # 学习率翻倍（大 batch size）
DEVICE="cuda:0"
SAVE_DIR="./outputs/rail"

# 数据采样参数（默认使用 10% 数据，约 2940 张图像）
SAMPLE_RATIO=0.1

# 预加载参数
PRELOAD_WORKERS=32    # 128 核 CPU 可以用更多 workers

# 创建输出目录
mkdir -p "$SAVE_DIR"

# 开始训练（启用预加载）
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
    --num_workers 8 \
    --use_patch True \
    --patch_size 900 \
    --patch_stride 850 \
    --depth_norm zscore \
    --train_sample_ratio $SAMPLE_RATIO \
    --preload \
    --preload_workers $PRELOAD_WORKERS

echo "========================================"
echo "Training finished!"
echo "========================================"
