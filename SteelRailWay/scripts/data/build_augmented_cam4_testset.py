#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build an augmented Cam4 test root with additional normal training images."""

# >>> path-bootstrap >>>
import os
import sys
from pathlib import Path

_PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))
# <<< path-bootstrap <<<

import argparse
import csv
import json
import shutil
from datetime import datetime
import cv2
import numpy as np


def build_parser():
    parser = argparse.ArgumentParser(
        description="Create rail_mvtec_gt_test_aug_cam4_normal50 without touching the original test root."
    )
    parser.add_argument("--test_root", type=str, default="rail_mvtec_gt_test")
    parser.add_argument("--normal_source_root", type=str, default="data_20260327")
    parser.add_argument("--out_root", type=str, default="rail_mvtec_gt_test_aug_cam4_normal50")
    parser.add_argument("--view_id", type=int, default=4)
    parser.add_argument("--target_good", type=int, default=50)
    parser.add_argument("--sampling_mode", type=str, default="uniform_time", choices=["uniform_time", "random"])
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--copy_mode", type=str, default="copy", choices=["copy", "hardlink", "symlink"])
    parser.add_argument("--patch_size", type=int, default=900)
    parser.add_argument("--patch_stride", type=int, default=850)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--depth_norm", type=str, default="zscore", choices=["zscore", "minmax", "log"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip_dataset_validate", action="store_true")
    return parser


def copy_one(src: Path, dst: Path, mode: str) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "symlink":
        os.symlink(src, dst)
    else:
        raise ValueError(f"unknown copy mode: {mode}")
    return str(dst)


def copy_tree(src_root: Path, out_root: Path, mode: str) -> None:
    def _copy(src, dst):
        return copy_one(Path(src), Path(dst), mode)

    shutil.copytree(src_root, out_root, copy_function=_copy)


def count_files(path: Path, suffix: str) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.iterdir() if p.is_file() and p.suffix.lower() == suffix)


def existing_frame_ids(root: Path, view_id: int) -> set[str]:
    frame_ids = set()
    for label_name in ["good", "broken"]:
        rgb_dir = root / "rail_mvtec" / f"cam{view_id}" / "test" / label_name
        if not rgb_dir.exists():
            continue
        for path in rgb_dir.glob("*.jpg"):
            frame_ids.add(path.stem)
    return frame_ids


def resolve_train_pairs(source_root: Path, view_id: int, patch_size: int, existing_ids: set[str]) -> list[dict]:
    rgb_dir = source_root / f"Cam{view_id}" / "rgb"
    depth_dir = source_root / f"Cam{view_id}" / "depth"
    if not rgb_dir.is_dir():
        raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")
    if not depth_dir.is_dir():
        raise FileNotFoundError(f"Depth directory not found: {depth_dir}")

    depth_names = {p.name for p in depth_dir.iterdir() if p.is_file()}
    rgb_files = sorted([p for p in rgb_dir.glob("*.jpg")])
    if not rgb_files:
        raise RuntimeError(f"No RGB jpg files found under {rgb_dir}")

    first = cv2.imread(str(rgb_files[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Cannot read first source image: {rgb_files[0]}")
    if first.shape[0] < patch_size:
        raise RuntimeError(
            f"normal_source_root looks pre-resized: first image height={first.shape[0]} < "
            f"patch_size={patch_size}. Use raw/full-height Cam{view_id} training data for patch eval."
        )

    pairs = []
    skipped_missing_depth = 0
    skipped_collision = 0
    for rgb_path in rgb_files:
        source_frame_id = rgb_path.stem
        target_frame_id = (
            source_frame_id[: -len("_reflectance")]
            if source_frame_id.endswith("_reflectance")
            else source_frame_id
        )
        if target_frame_id in existing_ids:
            skipped_collision += 1
            continue

        depth_name = target_frame_id + ".tiff"
        if depth_name not in depth_names:
            alt = target_frame_id + ".tif"
            if alt in depth_names:
                depth_name = alt
            else:
                skipped_missing_depth += 1
                continue

        pairs.append({
            "source_frame_id": source_frame_id,
            "target_frame_id": target_frame_id,
            "rgb_path": rgb_path,
            "depth_path": depth_dir / depth_name,
        })

    if skipped_missing_depth:
        print(f"[Augment] skipped {skipped_missing_depth} source RGB files without matched depth")
    if skipped_collision:
        print(f"[Augment] skipped {skipped_collision} source files colliding with existing test frame ids")
    return pairs


def sample_pairs(pairs: list[dict], need: int, mode: str, seed: int) -> list[dict]:
    if need <= 0:
        return []
    if len(pairs) < need:
        raise RuntimeError(f"Need {need} additional normal pairs, but only found {len(pairs)} candidates")

    if mode == "uniform_time":
        idx = np.linspace(0, len(pairs) - 1, num=need).round().astype(int)
        idx = np.unique(idx)
        if idx.size < need:
            pool = np.setdiff1d(np.arange(len(pairs)), idx)
            rng = np.random.default_rng(seed)
            extra = rng.choice(pool, size=need - idx.size, replace=False)
            idx = np.sort(np.concatenate([idx, extra]))
    else:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(pairs), size=need, replace=False))
    return [pairs[int(i)] for i in idx]


def validate_selected_pairs(selected: list[dict], patch_size: int) -> None:
    bad = []
    for item in selected:
        rgb = cv2.imread(str(item["rgb_path"]), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(item["depth_path"]), cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None:
            bad.append((item["target_frame_id"], "unreadable"))
            continue
        if rgb.shape[0] < patch_size or depth.shape[0] < patch_size:
            bad.append((item["target_frame_id"], f"rgb={rgb.shape[:2]}, depth={depth.shape[:2]}"))
    if bad:
        preview = "; ".join([f"{frame_id} ({reason})" for frame_id, reason in bad[:5]])
        raise RuntimeError(f"Selected source images are too small for patch_size={patch_size}: {preview}")


def collect_manifest(out_root: Path, test_root: Path, view_id: int, added: dict[str, dict]) -> list[dict]:
    rows = []
    for label_name, label in [("good", 0), ("broken", 1)]:
        rgb_dir = out_root / "rail_mvtec" / f"cam{view_id}" / "test" / label_name
        depth_dir = out_root / "rail_mvtec_depth" / f"cam{view_id}" / "test" / label_name
        if not rgb_dir.exists():
            continue
        for rgb_path in sorted(rgb_dir.glob("*.jpg")):
            frame_id = rgb_path.stem
            depth_path = depth_dir / f"{frame_id}.tiff"
            added_info = added.get(frame_id)
            if added_info:
                source = "train_normal"
                source_rgb = str(added_info["rgb_path"])
                source_depth = str(added_info["depth_path"])
                is_added = True
            else:
                source = "original_test"
                source_rgb = str(test_root / "rail_mvtec" / f"cam{view_id}" / "test" / label_name / rgb_path.name)
                source_depth = str(test_root / "rail_mvtec_depth" / f"cam{view_id}" / "test" / label_name / depth_path.name)
                is_added = False
            rows.append({
                "view_id": view_id,
                "frame_id": frame_id,
                "label_name": label_name,
                "label": label,
                "is_added": is_added,
                "source": source,
                "rgb_path": str(rgb_path),
                "depth_path": str(depth_path),
                "source_rgb_path": source_rgb,
                "source_depth_path": source_depth,
            })
    return rows


def write_manifest(out_root: Path, rows: list[dict], summary: dict) -> None:
    csv_path = out_root / "cam4_augmented_manifest.csv"
    json_path = out_root / "cam4_augmented_manifest.json"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "view_id", "frame_id", "label_name", "label", "is_added", "source",
            "rgb_path", "depth_path", "source_rgb_path", "source_depth_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "items": rows}, f, ensure_ascii=False, indent=2)
    print(f"[Augment] manifest CSV: {csv_path}")
    print(f"[Augment] manifest JSON: {json_path}")


def main():
    args = build_parser().parse_args()
    test_root = Path(args.test_root)
    source_root = Path(args.normal_source_root)
    out_root = Path(args.out_root)

    if not test_root.is_dir():
        raise FileNotFoundError(f"test_root not found: {test_root}")
    if out_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"out_root already exists: {out_root}. Pass --overwrite to rebuild it.")
        shutil.rmtree(out_root)

    original_good_dir = test_root / "rail_mvtec" / f"cam{args.view_id}" / "test" / "good"
    original_broken_dir = test_root / "rail_mvtec" / f"cam{args.view_id}" / "test" / "broken"
    original_good = count_files(original_good_dir, ".jpg")
    original_broken = count_files(original_broken_dir, ".jpg")
    if original_good == 0:
        raise RuntimeError(f"No original good images found under {original_good_dir}")
    if args.target_good < original_good:
        raise RuntimeError(f"target_good={args.target_good} is smaller than original good count={original_good}")

    print(f"[Augment] copying original test root: {test_root} -> {out_root}")
    copy_tree(test_root, out_root, args.copy_mode)

    needed = args.target_good - original_good
    existing_ids = existing_frame_ids(out_root, args.view_id)
    candidates = resolve_train_pairs(source_root, args.view_id, args.patch_size, existing_ids)
    selected = sample_pairs(candidates, needed, args.sampling_mode, args.random_seed)
    validate_selected_pairs(selected, args.patch_size)

    out_good_rgb_dir = out_root / "rail_mvtec" / f"cam{args.view_id}" / "test" / "good"
    out_good_depth_dir = out_root / "rail_mvtec_depth" / f"cam{args.view_id}" / "test" / "good"
    added = {}
    print(f"[Augment] adding {len(selected)} Cam{args.view_id} training normal pairs")
    for item in selected:
        frame_id = item["target_frame_id"]
        dst_rgb = out_good_rgb_dir / f"{frame_id}.jpg"
        dst_depth = out_good_depth_dir / f"{frame_id}.tiff"
        if dst_rgb.exists() or dst_depth.exists():
            raise FileExistsError(f"target frame already exists: {frame_id}")
        copy_one(item["rgb_path"], dst_rgb, args.copy_mode)
        copy_one(item["depth_path"], dst_depth, args.copy_mode)
        added[frame_id] = item

    final_good = count_files(out_good_rgb_dir, ".jpg")
    final_broken = count_files(out_root / "rail_mvtec" / f"cam{args.view_id}" / "test" / "broken", ".jpg")
    rows = collect_manifest(out_root, test_root, args.view_id, added)
    summary = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "test_root": str(test_root),
        "normal_source_root": str(source_root),
        "out_root": str(out_root),
        "view_id": args.view_id,
        "original_good": original_good,
        "original_broken": original_broken,
        "added_good": len(selected),
        "final_good": final_good,
        "final_broken": final_broken,
        "target_good": args.target_good,
        "sampling_mode": args.sampling_mode,
        "copy_mode": args.copy_mode,
    }
    write_manifest(out_root, rows, summary)

    if final_good != args.target_good:
        raise RuntimeError(f"Expected final good={args.target_good}, got {final_good}")

    if not args.skip_dataset_validate:
        from datasets.rail_dataset import RailDualModalDataset

        dataset = RailDualModalDataset(
            train_root=str(source_root),
            test_root=str(out_root),
            view_id=args.view_id,
            split="test",
            img_size=args.img_size,
            depth_norm=args.depth_norm,
            use_patch=True,
            patch_size=args.patch_size,
            patch_stride=args.patch_stride,
        )
        labels = [int(s["label"]) for s in dataset.samples]
        good = int(sum(label == 0 for label in labels))
        broken = int(sum(label == 1 for label in labels))
        print(
            f"[Augment] dataset validation: images={len(dataset.samples)}, "
            f"good={good}, broken={broken}, patches={len(dataset)}"
        )
        if good != args.target_good or broken != original_broken:
            raise RuntimeError(
                f"Dataset validation mismatch: good={good}, broken={broken}, "
                f"expected good={args.target_good}, broken={original_broken}"
            )

    print("[Augment] done")
    print(f"[Augment] original Cam{args.view_id}: good={original_good}, broken={original_broken}")
    print(f"[Augment] augmented Cam{args.view_id}: good={final_good}, broken={final_broken}")
    print(f"[Augment] out_root: {out_root}")


if __name__ == "__main__":
    main()
