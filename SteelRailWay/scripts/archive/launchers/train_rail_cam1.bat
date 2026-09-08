@echo off
REM 钢轨数据集 TRD 训练启动脚本 (Windows)

echo ========================================
echo Starting TRD Training on Rail Dataset
echo ========================================

REM 设置参数
set TRAIN_ROOT=G:/SteelRailWay/data_20260327
set TEST_ROOT=G:/SteelRailWay/rail_mvtec_gt_test
set VIEW_ID=1
set BATCH_SIZE=16
set EPOCHS=100
set LR=0.0001
set DEVICE=cuda:0
set SAVE_DIR=./outputs/rail

REM 数据采样参数（默认使用全部数据，本地小数据集）
set SAMPLE_RATIO=1.0

REM 创建输出目录
if not exist "%SAVE_DIR%" mkdir "%SAVE_DIR%"

REM 开始训练
python train/train_trd_rail.py ^
    --train_root %TRAIN_ROOT% ^
    --test_root %TEST_ROOT% ^
    --view_id %VIEW_ID% ^
    --batch_size %BATCH_SIZE% ^
    --epochs %EPOCHS% ^
    --lr %LR% ^
    --device %DEVICE% ^
    --save_dir %SAVE_DIR% ^
    --eval_interval 5 ^
    --num_workers 4 ^
    --use_patch True ^
    --patch_size 900 ^
    --patch_stride 850 ^
    --depth_norm zscore ^
    --train_sample_ratio %SAMPLE_RATIO%

echo ========================================
echo Training finished!
echo ========================================
pause
