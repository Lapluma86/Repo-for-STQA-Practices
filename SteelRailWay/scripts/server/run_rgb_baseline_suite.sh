#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_ROOT="${TRAIN_ROOT:-/data1/Leaddo_data/20260327-resize512}"
TEST_ROOT="${TEST_ROOT:-/home/root123/LF/WYM/SteelRailWay/rail_mvtec_gt_test}"
DEVICE="${DEVICE:-cuda:1}"
VIEW_ID="${VIEW_ID:-4}"
METHOD="${METHOD:-all}"
PRECISION="${PRECISION:-bf16}"
TRAIN_SAMPLE_NUM="${TRAIN_SAMPLE_NUM:-1500}"
TRAIN_VAL_TEST_SPLIT="${TRAIN_VAL_TEST_SPLIT:-0.9 0.1 0.0}"
SAMPLING_MODE="${SAMPLING_MODE:-uniform_time}"
IMG_SIZE="${IMG_SIZE:-512}"
PATCH_SIZE="${PATCH_SIZE:-900}"
PATCH_STRIDE="${PATCH_STRIDE:-850}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PRELOAD_WORKERS="${PRELOAD_WORKERS:-16}"
OUT_ROOT="${OUT_ROOT:-results/baselines_rgb/cam${VIEW_ID}}"
LOG_DIR="${LOG_DIR:-logs/baselines_rgb}"

mkdir -p "$OUT_ROOT" "$LOG_DIR"
LOG_FILE="$LOG_DIR/run_rgb_baseline_suite_cam${VIEW_ID}_$(date +%Y%m%d_%H%M%S).log"

echo "[INFO] TRAIN_ROOT=$TRAIN_ROOT" | tee "$LOG_FILE"
echo "[INFO] TEST_ROOT=$TEST_ROOT" | tee -a "$LOG_FILE"
echo "[INFO] METHOD=$METHOD" | tee -a "$LOG_FILE"
echo "[INFO] VIEW_ID=$VIEW_ID" | tee -a "$LOG_FILE"
echo "[INFO] sample plan: ${TRAIN_SAMPLE_NUM} -> 1350 train / 150 val" | tee -a "$LOG_FILE"
echo "[INFO] img_size=$IMG_SIZE" | tee -a "$LOG_FILE"
echo "[INFO] sampling_mode=$SAMPLING_MODE" | tee -a "$LOG_FILE"
echo "[INFO] test patch=${PATCH_SIZE}/${PATCH_STRIDE}" | tee -a "$LOG_FILE"

"$PYTHON_BIN" scripts/baselines/run_rgb_baseline.py \
  --method "$METHOD" \
  --train_root "$TRAIN_ROOT" \
  --test_root "$TEST_ROOT" \
  --view_id "$VIEW_ID" \
  --img_size "$IMG_SIZE" \
  --train_sample_num "$TRAIN_SAMPLE_NUM" \
  --train_val_test_split $TRAIN_VAL_TEST_SPLIT \
  --sampling_mode "$SAMPLING_MODE" \
  --patch_size "$PATCH_SIZE" \
  --patch_stride "$PATCH_STRIDE" \
  --device "$DEVICE" \
  --precision "$PRECISION" \
  --num_workers "$NUM_WORKERS" \
  --preload_workers "$PRELOAD_WORKERS" \
  --summary_csv "$OUT_ROOT/summary.csv" \
  --summary_json "$OUT_ROOT/summary.json" \
  2>&1 | tee -a "$LOG_FILE"

echo "[DONE] summary_csv=$OUT_ROOT/summary.csv" | tee -a "$LOG_FILE"
echo "[DONE] summary_json=$OUT_ROOT/summary.json" | tee -a "$LOG_FILE"
