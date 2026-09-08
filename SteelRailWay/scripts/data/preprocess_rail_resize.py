# -*- coding: utf-8 -*-
"""
钢轨数据集离线 resize 预处理（速度优化关键步骤）。

把 data_20260327/Cam{1-8}/{rgb,depth}/ 下的原始大图（6000×~900）一次性
resize 成正方形并落盘，训练时直接读小图，避免每 epoch 重复 decode + resize。

约定：
    输入: <src_root>/Cam{view}/rgb/*.jpg   (uint8, BGR)
          <src_root>/Cam{view}/depth/*.tiff (uint16)
    输出: <dst_root>/Cam{view}/rgb/*.jpg   (resize 后, 高质量 JPEG)
          <dst_root>/Cam{view}/depth/*.tiff (resize 后, 保持 uint16)

用法:
    python scripts/data/preprocess_rail_resize.py \
        --src_root /data/data_20260327 \
        --dst_root /data/data_20260327_512 \
        --size 512 --workers 16
"""

import os
import argparse
from concurrent.futures import ProcessPoolExecutor

import cv2
from tqdm import tqdm


def process_one(task):
    """处理单个文件：resize 后写盘。"""
    src, dst, size, is_depth = task
    if os.path.exists(dst):
        return  # 已存在则跳过，便于断点续跑

    if is_depth:
        # 深度图：保持 uint16，用 NEAREST 避免插值出非法深度值
        img = cv2.imread(src, cv2.IMREAD_UNCHANGED)
        if img is None:
            return
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(dst, img)
    else:
        # RGB：INTER_AREA 是缩小图像的最佳插值方式
        img = cv2.imread(src, cv2.IMREAD_COLOR)
        if img is None:
            return
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        cv2.imwrite(dst, img, [cv2.IMWRITE_JPEG_QUALITY, 95])


def collect_tasks(src_root, dst_root, size):
    tasks = []
    for cam in sorted(os.listdir(src_root)):
        cam_src = os.path.join(src_root, cam)
        if not (cam.startswith("Cam") and os.path.isdir(cam_src)):
            continue

        for sub_name, ext, is_depth in [("rgb", ".jpg", False), ("depth", ".tiff", True)]:
            src_dir = os.path.join(cam_src, sub_name)
            dst_dir = os.path.join(dst_root, cam, sub_name)
            if not os.path.isdir(src_dir):
                continue
            os.makedirs(dst_dir, exist_ok=True)
            for fname in os.listdir(src_dir):
                if fname.endswith(ext):
                    tasks.append((
                        os.path.join(src_dir, fname),
                        os.path.join(dst_dir, fname),
                        size,
                        is_depth,
                    ))
    return tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_root", required=True, help="原始数据根目录 (data_20260327)")
    parser.add_argument("--dst_root", required=True, help="输出根目录")
    parser.add_argument("--size", type=int, default=512, help="目标边长，与训练 img_size 对齐")
    parser.add_argument("--workers", type=int, default=16, help="并行进程数")
    args = parser.parse_args()

    tasks = collect_tasks(args.src_root, args.dst_root, args.size)
    print(f"[Preprocess] Total {len(tasks)} files to resize -> {args.dst_root}")

    if not tasks:
        print("[Preprocess] Nothing to do.")
        return

    # chunksize=8 减少进程间通信开销
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        list(tqdm(
            ex.map(process_one, tasks, chunksize=8),
            total=len(tasks),
            ncols=80,
            desc="Resize",
        ))

    print(f"[Preprocess] Done. Output: {args.dst_root}")


if __name__ == "__main__":
    main()
