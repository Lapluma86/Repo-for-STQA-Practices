#!/usr/bin/env bash
# ============================================================
# ADODAS 2026 训练启动脚本
# ============================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# ---- 默认配置 ----
TASK="a2"
PRESET="default"
GPU="0"
LUPI=""
EXTRA_ARGS=()

# ---- 预设配置 ----
declare -A PRESET_CONFIG
PRESET_CONFIG["default_a1"]="tasks/a1/default.yaml"
PRESET_CONFIG["default_a2"]="tasks/a2/default.yaml"
PRESET_CONFIG["full_a1"]="tasks/a1/mtl_full.yaml"
PRESET_CONFIG["full_a2"]="tasks/a2/mtl_full.yaml"

usage() {
    cat << 'EOF'
用法: ./run_train.sh [选项]

选项:
  --task a1|a2         任务 (默认: a2)
  --preset name         配置预设:
                          default   - 基线配置
                          full      - 全量优化 (MTL + 增强损失, LUPI 默认关闭)
                          debug     - 快速调试 (小模型, 2 epochs)
  --lupi mode           LUPI 模式:
                          heads     - 辅助属性预测头
                          weight    - 样本一致性加权
                          both      - 两者同时启用
  --gpu N               GPU ID (默认: 0)
  --name <run_name>     自定义运行名称 (默认: 自动生成)
  --extra "..."         传递给 train.py 的额外参数 (用于控制数据加载、跨模态注意力等)
  -h, --help            显示此帮助

环境变量:
  CUDA_VISIBLE_DEVICES  覆盖 GPU 选择
  ADODAS_DATA_ROOT      数据根目录 (默认: /data1/AdoDas)
  ADODAS_OUTPUT_ROOT    输出根目录 (默认: /data1/AdoDas/output)

示例:
  ./run_train.sh --task a2 --preset default                   # A2 基线训练
  ./run_train.sh --task a2 --preset full --gpu 0              # A2 全量优化
  ./run_train.sh --task a2 --preset default --lupi heads      # A2 + LUPI aux_heads
  ./run_train.sh --task a2 --preset default --lupi both       # A2 + LUPI 全部
  ./run_train.sh --task a2 --preset full --lupi weight        # A2 全量 + sample_reweight
  ./run_train.sh --task a2 --preset debug --gpu 1             # A2 快速调试

数据加载 (通过 --extra 控制):
  # HDF5 全量预载（推荐：稳定快速）
  --extra "--use_hdf5 1 --preload 1"
  # HDF5 按需加载（低内存，mmap 读盘）
  --extra "--use_hdf5 1 --preload 0"
  # Raw 文件预载（控制并行数防死锁）
  --extra "--preload 1 --preload_workers 4"

跨模态注意力 (P1):
  # 开启
  --extra "--use_cross_modal 1"
EOF
    exit 0
}

# ---- 参数解析 ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)
            TASK="$2"; shift 2 ;;
        --preset)
            PRESET="$2"; shift 2 ;;
        --lupi)
            LUPI="$2"; shift 2 ;;
        --gpu)
            GPU="$2"; shift 2 ;;
        --name)
            EXTRA_ARGS+=("--run_name" "$2"); shift 2 ;;
        --extra)
            EXTRA_ARGS+=($2); shift 2 ;;
        -h|--help)
            usage ;;
        *)
            echo "未知参数: $1"; usage ;;
    esac
done

# ---- 校验任务 ----
if [[ "$TASK" != "a1" && "$TASK" != "a2" ]]; then
    echo "错误: --task 必须是 a1 或 a2，得到: $TASK"
    exit 1
fi

# ---- 解析 LUPI 模式 ----
case "$LUPI" in
    heads)
        EXTRA_ARGS+=(--aux_lupi_enabled 1 --aux_lupi_heads 1)
        ;;
    weight)
        EXTRA_ARGS+=(--aux_lupi_enabled 1 --aux_lupi_reweight 1)
        ;;
    both)
        EXTRA_ARGS+=(--aux_lupi_enabled 1 --aux_lupi_heads 1 --aux_lupi_reweight 1)
        ;;
    "")
        ;;
    *)
        echo "错误: --lupi 必须是 heads / weight / both，得到: $LUPI"
        exit 1
        ;;
esac

# ---- 解析预设 ----
CONFIG=""
case "$PRESET" in
    default)
        CONFIG="${PRESET_CONFIG["default_$TASK"]}" ;;
    full)
        CONFIG="${PRESET_CONFIG["full_$TASK"]}" ;;
    debug)
        # 快速调试：用 default config + 覆盖参数
        CONFIG="${PRESET_CONFIG["default_$TASK"]}"
        EXTRA_ARGS+=(
            --epochs 2 --batch_size 8 --tcn_layers 2 --d_model 128 --d_shared 128
            --num_workers 4 --preload 0 --max_participants 100
        )
        ;;
    *)
        # 尝试直接作为配置文件路径
        if [[ -f "$PRESET" ]]; then
            CONFIG="$PRESET"
        else
            echo "错误: 未知预设 '$PRESET'，且不是有效的配置文件路径"
            echo "可用预设: default, full, debug"
            exit 1
        fi
        ;;
esac

# ---- 验证关键路径 ----
DATA_ROOT="${ADODAS_DATA_ROOT:-/data1/AdoDas}"
OUTPUT_ROOT="${ADODAS_OUTPUT_ROOT:-./output}"

if [[ ! -d "$DATA_ROOT" ]]; then
    echo "错误: 数据根目录不存在: $DATA_ROOT"
    echo "可通过 ADODAS_DATA_ROOT 环境变量指定"
    exit 1
fi

if [[ ! -f "$CONFIG" ]]; then
    echo "错误: 配置文件不存在: $CONFIG"
    exit 1
fi

CONDA_ENV="${ADODAS_CONDA_ENV:-adodas}"

# ---- 打印启动信息 ----
echo "============================================================"
echo " ADODAS 2026 Training Launcher"
echo "============================================================"
echo "  Task:       $TASK"
echo "  Preset:     $PRESET"
echo "  Config:     $CONFIG"
echo "  GPU:        $GPU"
echo "  LUPI:       ${LUPI:-none}"
echo "  Data root:  $DATA_ROOT"
echo "  Output:     $OUTPUT_ROOT"
echo "  Conda env:  $CONDA_ENV"
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    echo "  Extra:      ${EXTRA_ARGS[*]}"
fi
echo "============================================================"

# ---- 启动训练 ----
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
    if conda env list | grep -q "^${CONDA_ENV} "; then
        conda activate "$CONDA_ENV"
        echo "  Conda env '$CONDA_ENV' activated"
    else
        echo "  警告: conda 环境 '$CONDA_ENV' 未找到，使用当前 Python"
    fi
fi

echo "  Running: python train.py --task $TASK --config $CONFIG ${EXTRA_ARGS[*]}"
echo "============================================================"
echo ""

exec python train.py \
    --task "$TASK" \
    --config "$CONFIG" \
    --feature_root "$DATA_ROOT" \
    --manifest_dir "$DATA_ROOT" \
    --output_dir "$OUTPUT_ROOT" \
    "${EXTRA_ARGS[@]}"
