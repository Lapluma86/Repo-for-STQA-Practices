#!/bin/bash
# 钢轨数据集 TRD 训练启动脚本

# 设置 Python 路径
export PYTHONPATH="G:/SteelRailWay:$PYTHONPATH"

# 训练参数
TRAIN_ROOT="G:/SteelRailWay/data_20260327"
TEST_ROOT="G:/SteelRailWay/rail_mvtec_gt_test"
VIEW_ID=1
BATCH_SIZE=16
EPOCHS=100
LR=0.0001
DEVICE="cuda:0"
SAVE_DIR="./outputs/rail"

# 创建输出目录
mkdir -p $SAVE_DIR

# 开始训练
python train/train_trd_rail.py \
    --train_root $TRAIN_ROOT \
    --test_root $TEST_ROOT \
    --view_id $VIEW_ID \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --device $DEVICE \
    --save_dir $SAVE_DIR \
    --eval_interval 5 \
    --num_workers 4 \
    --use_patch True \
    --patch_size 900 \
    --patch_stride 850 \
    --depth_norm zscore

echo "Training finished!"
