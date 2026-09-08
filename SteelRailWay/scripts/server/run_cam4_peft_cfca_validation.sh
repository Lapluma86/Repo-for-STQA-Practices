#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
TRAIN_ROOT="${TRAIN_ROOT:-/data1/Leaddo_data/20260327-resize512}"
TEST_ROOT="${TEST_ROOT:-/home/root123/LF/WYM/SteelRailWay/rail_mvtec_gt_test}"
BASELINE_CKPT="${BASELINE_CKPT:-outputs/rail_all/Cam4/20260501_014226_cam4_bs32_lr0.005_img512_ratio1.0/best_cam4.pth}"
PEFT_CKPT="${PEFT_CKPT:-outputs/rail_peft/cam4_p1_20260501_225618/final/final_peft_cam4.pth}"
REFERENCE_STATS="${REFERENCE_STATS:-outputs/rail_peft/cam4_p1_20260501_225618/stats/reference_stats.pt}"

OUT_ROOT="${OUT_ROOT:-outputs/rail_ablation}"
PEFT_ABLATION_OUT="${PEFT_ABLATION_OUT:-$OUT_ROOT/cam4_cf_ca_peft}"
FUSION_RULE_OUT="${FUSION_RULE_OUT:-$OUT_ROOT/cam4_cf_ca_peft_fusion_rules}"
AUDIT_OUT="${AUDIT_OUT:-$OUT_ROOT/cam4_peft_scope_audit}"
REPAIR_OUT="${REPAIR_OUT:-$OUT_ROOT/cam4_cfca_repair}"
CASE_GROUP_OUT="${CASE_GROUP_OUT:-$OUT_ROOT/cam4_case_groups}"

PRECISION="${PRECISION:-fp32}"
DEPTH_NORM="${DEPTH_NORM:-zscore}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PATCH_SIZE="${PATCH_SIZE:-900}"
PATCH_STRIDE="${PATCH_STRIDE:-850}"

REPAIR_EPOCHS="${REPAIR_EPOCHS:-100}"
REPAIR_LR="${REPAIR_LR:-1e-4}"
REPAIR_BATCH_SIZE="${REPAIR_BATCH_SIZE:-8}"
REPAIR_EVAL_BATCH_SIZE="${REPAIR_EVAL_BATCH_SIZE:-2}"
REPAIR_NUM_WORKERS="${REPAIR_NUM_WORKERS:-4}"
REPAIR_SCOPES="${REPAIR_SCOPES:-cf_only ca_only cf_ca}"

BOOTSTRAP_ITERS="${BOOTSTRAP_ITERS:-10000}"

RUN_AUDIT="${RUN_AUDIT:-1}"
RUN_POSTHOC_ABLATION="${RUN_POSTHOC_ABLATION:-1}"
RUN_FUSION_RULES="${RUN_FUSION_RULES:-1}"
RUN_BOOTSTRAP="${RUN_BOOTSTRAP:-1}"
RUN_REPAIR="${RUN_REPAIR:-1}"
RUN_CASE_ANALYSIS="${RUN_CASE_ANALYSIS:-0}"

echo "[INFO] ROOT_DIR=$ROOT_DIR"
echo "[INFO] PYTHON_BIN=$PYTHON_BIN"
echo "[INFO] DEVICE=$DEVICE"

mkdir -p "$OUT_ROOT"

if [[ "$RUN_AUDIT" == "1" ]]; then
  echo "[STEP] V1 audit: trainable scope and CF/CA parameter diff"
  "$PYTHON_BIN" scripts/diagnostics/audit_cam4_peft_scope.py \
    --baseline_ckpt "$BASELINE_CKPT" \
    --peft_ckpt "$PEFT_CKPT" \
    --reference_stats "$REFERENCE_STATS" \
    --train_root "$TRAIN_ROOT" \
    --test_root "$TEST_ROOT" \
    --view_id 4 \
    --img_size 512 \
    --depth_norm "$DEPTH_NORM" \
    --batch_size 4 \
    --num_workers "$NUM_WORKERS" \
    --device "$DEVICE" \
    --precision "$PRECISION" \
    --out_dir "$AUDIT_OUT"
fi

if [[ "$RUN_POSTHOC_ABLATION" == "1" ]]; then
  echo "[STEP] Post-hoc PEFT CF/CA ablation"
  "$PYTHON_BIN" scripts/ablation/run_cam4_cf_ca.py \
    --ckpt "$BASELINE_CKPT" \
    --depth_peft_ckpt "$PEFT_CKPT" \
    --train_root "$TRAIN_ROOT" \
    --test_root "$TEST_ROOT" \
    --out_root "$PEFT_ABLATION_OUT" \
    --modes full no_cf no_ca no_cf_ca \
    --view_id 4 \
    --img_size 512 \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --device "$DEVICE" \
    --depth_norm "$DEPTH_NORM" \
    --precision "$PRECISION" \
    --fusion_rule sum \
    --patch_size "$PATCH_SIZE" \
    --patch_stride "$PATCH_STRIDE" \
    --assist_fill zeros
fi

if [[ "$RUN_FUSION_RULES" == "1" ]]; then
  echo "[STEP] V5 fusion-rule rerun"
  "$PYTHON_BIN" scripts/diagnostics/rerun_cam4_cfca_fusion_rules.py \
    --ckpt "$BASELINE_CKPT" \
    --depth_peft_ckpt "$PEFT_CKPT" \
    --train_root "$TRAIN_ROOT" \
    --test_root "$TEST_ROOT" \
    --view_id 4 \
    --img_size 512 \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --device "$DEVICE" \
    --precision "$PRECISION" \
    --depth_norm "$DEPTH_NORM" \
    --patch_size "$PATCH_SIZE" \
    --patch_stride "$PATCH_STRIDE" \
    --assist_fill zeros \
    --out_dir "$FUSION_RULE_OUT"
fi

if [[ "$RUN_BOOTSTRAP" == "1" ]]; then
  echo "[STEP] V3 bootstrap CI"
  "$PYTHON_BIN" scripts/diagnostics/bootstrap_cam4_cfca.py \
    --scores_root "$PEFT_ABLATION_OUT" \
    --iterations "$BOOTSTRAP_ITERS" \
    --seed 42
fi

if [[ "$RUN_REPAIR" == "1" ]]; then
  echo "[STEP] V2 CF/CA repair micro-tuning"
  "$PYTHON_BIN" scripts/train/train_cam4_peft_cfca_repair.py \
    --ckpt "$BASELINE_CKPT" \
    --depth_peft_ckpt "$PEFT_CKPT" \
    --train_root "$TRAIN_ROOT" \
    --test_root "$TEST_ROOT" \
    --view_id 4 \
    --img_size 512 \
    --depth_norm "$DEPTH_NORM" \
    --device "$DEVICE" \
    --precision "$PRECISION" \
    --epochs "$REPAIR_EPOCHS" \
    --lr "$REPAIR_LR" \
    --batch_size "$REPAIR_BATCH_SIZE" \
    --eval_batch_size "$REPAIR_EVAL_BATCH_SIZE" \
    --num_workers "$REPAIR_NUM_WORKERS" \
    --folds 4 \
    --log_every 10 \
    --patch_size "$PATCH_SIZE" \
    --patch_stride "$PATCH_STRIDE" \
    --scope $REPAIR_SCOPES \
    --output_root "$REPAIR_OUT"
fi

if [[ "$RUN_CASE_ANALYSIS" == "1" ]]; then
  echo "[STEP] V4 case-group analysis"
  "$PYTHON_BIN" scripts/diagnostics/analyze_cam4_case_groups.py \
    --scores_root "$PEFT_ABLATION_OUT" \
    --frame_groups "$CASE_GROUP_OUT/frame_groups.csv" \
    --out_dir "$CASE_GROUP_OUT/analysis"
fi

echo "[DONE] Cam4 PEFT CF/CA validation pipeline finished."
