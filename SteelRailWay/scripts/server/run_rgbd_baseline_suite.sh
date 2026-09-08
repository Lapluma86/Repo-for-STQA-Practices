#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_ROOT="${TRAIN_ROOT:-/data1/Leaddo_data/20260327-resize512}"
TEST_ROOT="${TEST_ROOT:-/home/root123/LF/WYM/SteelRailWay/rail_mvtec_gt_test}"
DEVICE="${DEVICE:-cuda:1}"
VIEW_IDS="${VIEW_IDS:-1 4 5}"
METHOD="${METHOD:-all}"
PRECISION="${PRECISION:-bf16}"
DEPTH_NORM="${DEPTH_NORM:-zscore}"
SCORE_SOURCE="${SCORE_SOURCE:-fusion}"
FUSION_RULE="${FUSION_RULE:-sum}"
TRAIN_SAMPLE_NUM="${TRAIN_SAMPLE_NUM:-1500}"
TRAIN_VAL_TEST_SPLIT="${TRAIN_VAL_TEST_SPLIT:-0.9 0.1 0.0}"
SAMPLING_MODE="${SAMPLING_MODE:-uniform_time}"
IMG_SIZE="${IMG_SIZE:-512}"
PATCH_SIZE="${PATCH_SIZE:-900}"
PATCH_STRIDE="${PATCH_STRIDE:-850}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PRELOAD_WORKERS="${PRELOAD_WORKERS:-16}"
OUT_ROOT="${OUT_ROOT:-results/baselines_rgbd}"
LOG_DIR="${LOG_DIR:-logs/baselines_rgbd}"

mkdir -p "$OUT_ROOT" "$LOG_DIR"
LOG_FILE="$LOG_DIR/run_rgbd_baseline_suite_$(date +%Y%m%d_%H%M%S).log"

echo "[INFO] TRAIN_ROOT=$TRAIN_ROOT" | tee "$LOG_FILE"
echo "[INFO] TEST_ROOT=$TEST_ROOT" | tee -a "$LOG_FILE"
echo "[INFO] METHOD=$METHOD" | tee -a "$LOG_FILE"
echo "[INFO] VIEW_IDS=$VIEW_IDS" | tee -a "$LOG_FILE"
echo "[INFO] sample plan: ${TRAIN_SAMPLE_NUM} -> 1350 train / 150 val" | tee -a "$LOG_FILE"
echo "[INFO] img_size=$IMG_SIZE" | tee -a "$LOG_FILE"
echo "[INFO] sampling_mode=$SAMPLING_MODE" | tee -a "$LOG_FILE"
echo "[INFO] depth_norm=$DEPTH_NORM" | tee -a "$LOG_FILE"
echo "[INFO] input_mode=rgbd" | tee -a "$LOG_FILE"
echo "[INFO] fusion_rule=$FUSION_RULE" | tee -a "$LOG_FILE"
echo "[INFO] test patch=${PATCH_SIZE}/${PATCH_STRIDE}" | tee -a "$LOG_FILE"

declare -a SUMMARY_PATHS=()

for VIEW_ID in $VIEW_IDS; do
  VIEW_OUT="$OUT_ROOT/cam${VIEW_ID}"
  mkdir -p "$VIEW_OUT"
  echo "[RUN] view_id=$VIEW_ID" | tee -a "$LOG_FILE"
  "$PYTHON_BIN" scripts/baselines/run_rgbd_baseline.py \
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
    --depth_norm "$DEPTH_NORM" \
    --score_source "$SCORE_SOURCE" \
    --fusion_rule "$FUSION_RULE" \
    --num_workers "$NUM_WORKERS" \
    --preload_workers "$PRELOAD_WORKERS" \
    --summary_csv "$VIEW_OUT/summary.csv" \
    --summary_json "$VIEW_OUT/summary.json" \
    2>&1 | tee -a "$LOG_FILE"
  SUMMARY_PATHS+=("$VIEW_OUT/summary.csv")
done

MERGED_CSV="$OUT_ROOT/summary_cam1_cam4_cam5.csv"
MERGED_JSON="$OUT_ROOT/summary_cam1_cam4_cam5.json"

"$PYTHON_BIN" - <<'PY' "$MERGED_CSV" "$MERGED_JSON" "${SUMMARY_PATHS[@]}"
import csv
import json
import sys
from pathlib import Path

merged_csv = Path(sys.argv[1])
merged_json = Path(sys.argv[2])
rows = []
fieldnames = []
for path_str in sys.argv[3:]:
    path = Path(path_str)
    if not path.exists():
        continue
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames:
            for name in reader.fieldnames:
                if name not in fieldnames:
                    fieldnames.append(name)
        rows.extend(list(reader))

merged_csv.parent.mkdir(parents=True, exist_ok=True)
with merged_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

with merged_json.open("w", encoding="utf-8") as f:
    json.dump({"rows": rows}, f, ensure_ascii=False, indent=2)
PY

echo "[DONE] merged_summary_csv=$MERGED_CSV" | tee -a "$LOG_FILE"
echo "[DONE] merged_summary_json=$MERGED_JSON" | tee -a "$LOG_FILE"
