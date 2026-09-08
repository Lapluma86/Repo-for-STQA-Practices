#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速测试脚本：验证 rail_dataset.py 的自动 pre-resize 检测 + 数据加载。

用法：
    python scripts/diagnostics/test_dataset_loading.py --train_root /data1/Leaddo_data/20260327-resize512 --view_id 1
"""

import sys
import os
from pathlib import Path
_PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

import argparse
import time
from datasets.rail_dataset import RailDualModalDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_root", type=str, required=True,
                        help="训练数据根目录（可以是原始或 resize 后的）")
    parser.add_argument("--test_root", type=str, default="./rail_mvtec_gt_test",
                        help="测试数据根目录")
    parser.add_argument("--view_id", type=int, default=1)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--use_patch", action="store_true", default=False)
    parser.add_argument("--sampling_mode", type=str, default="uniform_time")
    args = parser.parse_args()

    print("=" * 60)
    print("测试 RailDualModalDataset 数据加载")
    print("=" * 60)
    print(f"train_root: {args.train_root}")
    print(f"view_id:    {args.view_id}")
    print(f"img_size:   {args.img_size}")
    print(f"use_patch:  {args.use_patch}")
    print(f"sampling_mode: {args.sampling_mode}")
    print()

    # 创建训练集（不预加载，快速测试）
    print("[1/3] 创建训练集...")
    start = time.time()
    train_ds = RailDualModalDataset(
        train_root=args.train_root,
        test_root=args.test_root,
        view_id=args.view_id,
        split="train",
        img_size=args.img_size,
        use_patch=args.use_patch,
        train_sample_ratio=0.1,  # 只取 10% 快速测试
        sampling_mode=args.sampling_mode,
        preload=False,
    )
    print(f"   训练集大小: {len(train_ds)} 样本")
    print(f"   is_preresized: {train_ds.is_preresized}")
    print(f"   耗时: {time.time()-start:.2f}s\n")

    # 读取第一个样本
    print("[2/3] 读取第一个训练样本...")
    start = time.time()
    sample = train_ds[0]
    print(f"   intensity shape: {sample['intensity'].shape}")
    print(f"   depth shape:     {sample['depth'].shape}")
    print(f"   label:           {sample['label']}")
    print(f"   frame_id:        {sample['frame_id']}")
    print(f"   耗时: {time.time()-start:.2f}s\n")

    # 创建验证集
    print("[3/3] 创建验证集...")
    start = time.time()
    val_ds = RailDualModalDataset(
        train_root=args.train_root,
        test_root=args.test_root,
        view_id=args.view_id,
        split="val",
        img_size=args.img_size,
        use_patch=args.use_patch,
        train_sample_ratio=0.1,
        sampling_mode=args.sampling_mode,
        preload=False,
    )
    print(f"   验证集大小: {len(val_ds)} 样本")
    print(f"   is_preresized: {val_ds.is_preresized}")
    print(f"   耗时: {time.time()-start:.2f}s\n")

    print("=" * 60)
    print("✅ 数据加载测试通过")
    print("=" * 60)
    print()
    print("关键检查项：")
    print(f"  1. is_preresized = {train_ds.is_preresized}")
    if train_ds.is_preresized:
        print("     → 已检测到 pre-resized 数据，运行时跳过 resize ✅")
    else:
        print("     → 未检测到 pre-resized 标记，运行时会执行 resize")
    print(f"  2. 训练集样本数 = {len(train_ds)}")
    print(f"  3. 验证集样本数 = {len(val_ds)}")
    print(f"  4. 采样模式 = {args.sampling_mode}")
    print()
    print("如需完整训练，运行：")
    print(f"  python train/train_trd_rail.py --train_root {args.train_root} --view_id {args.view_id}")

if __name__ == "__main__":
    main()
