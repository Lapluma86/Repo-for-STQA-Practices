#!/bin/bash
# =============================================================================
# 钢轨 8 视角批量训练脚本（速度优先 + 质量保留）
#
# 使用前提：
#   1) 已用 scripts/data/preprocess_rail_resize.py 把数据 resize 到 512×512 落盘
#   2) RTX 6000 Pro / Ada / Ampere 等支持 bf16 的 GPU
#   3) 数据按 view 命名: <TRAIN_ROOT>/Cam{1..8}/{rgb,depth}/
#
# 速度提升来源（相对原方案）：
#   - 训练阶段关闭 patch 切分（每图样本量 7→1）
#   - bf16 混合精度（无 GradScaler 开销）
#   - channels_last 内存格式（Tensor Core 提速 10-30%）
#   - persistent_workers + prefetch_factor（避免 worker 反复初始化）
#   - 整库预加载到内存（消除 IO 瓶颈）
#   - batch=128（充分利用 80GB 显存）
#
# 质量保证：
#   - 测试阶段仍用 patch（小缺陷召回）
#   - 时间均匀采样（数据分布稳定）
#   - 仅保留每个视角的 best 模型
# =============================================================================

set -e
cd "$(dirname "$0")/../.."

# ============== 路径配置（按需修改） ==============
# 已经离线 resize 后的训练数据根目录
TRAIN_ROOT="${TRAIN_ROOT:-/data1/Leaddo_data/20260327-resize512}"
# 测试数据根目录（保持原始大图，patch 推理）
TEST_ROOT="${TEST_ROOT:-/home/root123/LF/WYM/SteelRailWay/rail_mvtec_gt_test}"
# 输出根目录（每个视角会写入 $SAVE_DIR/CamN/）
SAVE_DIR="${SAVE_DIR:-./outputs/rail_all}"

# ============== 训练超参（速度 + 质量平衡） ==============
IMG_SIZE=512
BATCH_SIZE=32
EPOCHS=200
LR=0.005
NUM_WORKERS=16
PRELOAD_WORKERS=32
PRECISION=bf16     # RTX 6000 Pro 首选；老卡可改 fp16
TRAIN_SAMPLE_NUM=1200
SAMPLING_MODE=uniform_time

# ============== 视角列表（默认全部 8 个） ==============
VIEWS="${VIEWS:-1 2 3 4 5 6 7 8}"

# ============== 单 GPU 顺序训练 ==============
DEVICE="${DEVICE:-cuda:0}"

mkdir -p "$SAVE_DIR"
SUMMARY_DIR="$SAVE_DIR/_summaries"
mkdir -p "$SUMMARY_DIR"
SUMMARY="$SUMMARY_DIR/summary_$(date +%Y%m%d_%H%M%S).txt"
echo "Training started at $(date)" | tee "$SUMMARY"
echo "Train root: $TRAIN_ROOT" | tee -a "$SUMMARY"
echo "Test root:  $TEST_ROOT" | tee -a "$SUMMARY"
echo "Views:      $VIEWS" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"

for v in $VIEWS; do
    CAM_SAVE_DIR="$SAVE_DIR/Cam${v}"
    mkdir -p "$CAM_SAVE_DIR"

    echo "==========================================" | tee -a "$SUMMARY"
    echo "  Training Cam${v}" | tee -a "$SUMMARY"
    echo "  Output: $CAM_SAVE_DIR" | tee -a "$SUMMARY"
    echo "==========================================" | tee -a "$SUMMARY"

    python train/train_trd_rail.py \
        --train_root "$TRAIN_ROOT" \
        --test_root  "$TEST_ROOT" \
        --view_id    $v \
        --img_size   $IMG_SIZE \
        --batch_size $BATCH_SIZE \
        --epochs     $EPOCHS \
        --lr         $LR \
        --num_workers $NUM_WORKERS \
        --preload \
        --preload_workers $PRELOAD_WORKERS \
        --precision  $PRECISION \
        --channels_last \
        --train_sample_num $TRAIN_SAMPLE_NUM \
        --sampling_mode $SAMPLING_MODE \
        --device     "$DEVICE" \
        --save_dir   "$CAM_SAVE_DIR" \
        2>&1 | tee -a "$SUMMARY"

    echo "" | tee -a "$SUMMARY"
done

echo "All views finished at $(date)" | tee -a "$SUMMARY"
echo "Models saved under: $SAVE_DIR/CamN" | tee -a "$SUMMARY"
