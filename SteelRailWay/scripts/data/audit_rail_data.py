#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Audit rail RGB/depth/GT data layout and basic image statistics.

This script is intentionally lightweight: it checks file pairing, dimensions,
mask validity, and simple RGB/depth statistics without loading any model.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


IMG_EXTS = {".jpg", ".jpeg", ".png"}
DEPTH_EXTS = {".tif", ".tiff"}


def frame_id(path: Path) -> str:
    return path.stem


def list_files(path: Path, exts: set[str]) -> list[Path]:
    if not path.exists():
        return []
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in exts)


def image_stats(path: Path, grayscale: bool = False) -> dict:
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_UNCHANGED
    img = cv2.imread(str(path), flag)
    if img is None:
        return {"read_ok": False}

    arr = img.astype(np.float64)
    if arr.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    else:
        gray = arr

    valid = gray[gray > 0]
    valid_ratio = float(valid.size / gray.size) if gray.size else 0.0
    if valid.size == 0:
        valid = gray.reshape(-1)

    return {
        "read_ok": True,
        "height": int(img.shape[0]),
        "width": int(img.shape[1]),
        "channels": int(img.shape[2]) if img.ndim == 3 else 1,
        "dtype": str(img.dtype),
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid)),
        "nonzero_ratio": valid_ratio,
    }


def audit_test(root: Path) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    issues: list[str] = []

    rgb_root = root / "rail_mvtec"
    depth_root = root / "rail_mvtec_depth"

    for cam_dir in sorted(rgb_root.glob("cam*")):
        if not cam_dir.is_dir():
            continue
        cam = cam_dir.name
        for label_name, label in [("good", 0), ("broken", 1)]:
            rgb_dir = rgb_root / cam / "test" / label_name
            depth_dir = depth_root / cam / "test" / label_name
            gt_dir = rgb_root / cam / "ground_truth" / "broken"
            depth_gt_dir = depth_root / cam / "ground_truth" / "broken"

            rgb_files = list_files(rgb_dir, IMG_EXTS)
            depth_files = list_files(depth_dir, DEPTH_EXTS)
            rgb_ids = {frame_id(p): p for p in rgb_files}
            depth_ids = {frame_id(p): p for p in depth_files}

            for fid in sorted(set(rgb_ids) | set(depth_ids)):
                rgb_path = rgb_ids.get(fid)
                depth_path = depth_ids.get(fid)
                gt_path = gt_dir / f"{fid}.png"
                depth_gt_path = depth_gt_dir / f"{fid}.png"

                row = {
                    "split": "test",
                    "cam": cam,
                    "label_name": label_name,
                    "label": label,
                    "frame_id": fid,
                    "rgb_path": str(rgb_path) if rgb_path else "",
                    "depth_path": str(depth_path) if depth_path else "",
                    "gt_path": str(gt_path) if label else "",
                    "depth_gt_path": str(depth_gt_path) if label else "",
                    "rgb_exists": bool(rgb_path),
                    "depth_exists": bool(depth_path),
                    "gt_exists": bool(label and gt_path.exists()),
                    "depth_gt_exists": bool(label and depth_gt_path.exists()),
                }

                if not rgb_path:
                    issues.append(f"{cam}/{label_name}/{fid}: missing RGB")
                if not depth_path:
                    issues.append(f"{cam}/{label_name}/{fid}: missing depth")
                if label and not gt_path.exists():
                    issues.append(f"{cam}/{label_name}/{fid}: missing RGB GT")
                if label and not depth_gt_path.exists():
                    issues.append(f"{cam}/{label_name}/{fid}: missing depth GT")

                if rgb_path:
                    row.update({f"rgb_{k}": v for k, v in image_stats(rgb_path).items()})
                if depth_path:
                    row.update({f"depth_{k}": v for k, v in image_stats(depth_path, grayscale=True).items()})
                if label and gt_path.exists():
                    gt_stats = image_stats(gt_path, grayscale=True)
                    row.update({f"gt_{k}": v for k, v in gt_stats.items()})
                    if gt_stats.get("read_ok") and gt_stats.get("nonzero_ratio", 0.0) <= 0.0:
                        issues.append(f"{cam}/{label_name}/{fid}: RGB GT mask is empty")
                if label and depth_gt_path.exists():
                    dgt_stats = image_stats(depth_gt_path, grayscale=True)
                    row.update({f"depth_gt_{k}": v for k, v in dgt_stats.items()})
                    if dgt_stats.get("read_ok") and dgt_stats.get("nonzero_ratio", 0.0) <= 0.0:
                        issues.append(f"{cam}/{label_name}/{fid}: depth GT mask is empty")

                if (
                    row.get("rgb_read_ok") and row.get("depth_read_ok")
                    and (row.get("rgb_height") != row.get("depth_height")
                         or row.get("rgb_width") != row.get("depth_width"))
                ):
                    issues.append(
                        f"{cam}/{label_name}/{fid}: RGB/depth shape mismatch "
                        f"{row.get('rgb_height')}x{row.get('rgb_width')} vs "
                        f"{row.get('depth_height')}x{row.get('depth_width')}"
                    )

                rows.append(row)

    return rows, issues


def pick_evenly(items: list[str], limit: int | None) -> list[str]:
    if limit is None or limit <= 0 or len(items) <= limit:
        return items
    idx = np.linspace(0, len(items) - 1, num=limit).round().astype(int)
    idx = np.unique(idx)
    return [items[int(i)] for i in idx]


def audit_train(root: Path, sample_limit: int | None = None) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    issues: list[str] = []

    for cam_dir in sorted(root.glob("Cam*")):
        if not cam_dir.is_dir():
            continue
        rgb_dir = cam_dir / "rgb"
        depth_dir = cam_dir / "depth"
        rgb_files = list_files(rgb_dir, IMG_EXTS)
        depth_files = list_files(depth_dir, DEPTH_EXTS)

        def normalize_depth_id(fid: str) -> str:
            return fid[:-len("_reflectance")] if fid.endswith("_reflectance") else fid

        rgb_ids = {normalize_depth_id(frame_id(p)): p for p in rgb_files}
        depth_ids = {frame_id(p): p for p in depth_files}
        sample_ids = pick_evenly(sorted(set(rgb_ids) | set(depth_ids)), sample_limit)

        for fid in sample_ids:
            rgb_path = rgb_ids.get(fid)
            depth_path = depth_ids.get(fid)
            row = {
                "split": "train",
                "cam": cam_dir.name,
                "label_name": "normal",
                "label": 0,
                "frame_id": fid,
                "rgb_path": str(rgb_path) if rgb_path else "",
                "depth_path": str(depth_path) if depth_path else "",
                "rgb_exists": bool(rgb_path),
                "depth_exists": bool(depth_path),
            }
            if not rgb_path:
                issues.append(f"{cam_dir.name}/{fid}: missing train RGB")
            if not depth_path:
                issues.append(f"{cam_dir.name}/{fid}: missing train depth")
            if rgb_path:
                row.update({f"rgb_{k}": v for k, v in image_stats(rgb_path).items()})
            if depth_path:
                row.update({f"depth_{k}": v for k, v in image_stats(depth_path, grayscale=True).items()})
            rows.append(row)

    return rows, issues


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row.get("split"), row.get("cam"), row.get("label_name"))].append(row)

    summary = []
    for key, group in sorted(groups.items()):
        split, cam, label_name = key

        def values(field: str) -> list[float]:
            return [
                float(r[field])
                for r in group
                if field in r and r[field] not in ("", None)
            ]

        item = {
            "split": split,
            "cam": cam,
            "label_name": label_name,
            "count": len(group),
            "rgb_missing": sum(1 for r in group if not r.get("rgb_exists")),
            "depth_missing": sum(1 for r in group if not r.get("depth_exists")),
            "gt_missing": sum(1 for r in group if r.get("label") == 1 and not r.get("gt_exists")),
            "rgb_height_min": min(values("rgb_height"), default=""),
            "rgb_height_max": max(values("rgb_height"), default=""),
            "rgb_width_min": min(values("rgb_width"), default=""),
            "rgb_width_max": max(values("rgb_width"), default=""),
            "rgb_mean_avg": np.mean(values("rgb_mean")).item() if values("rgb_mean") else "",
            "depth_mean_avg": np.mean(values("depth_mean")).item() if values("depth_mean") else "",
            "gt_nonzero_ratio_avg": (
                np.mean(values("gt_nonzero_ratio")).item()
                if values("gt_nonzero_ratio") else ""
            ),
        }
        summary.append(item)
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit rail dataset files and stats.")
    parser.add_argument("--test_root", type=str, default="rail_mvtec_gt_test")
    parser.add_argument("--train_root", type=str, default="data_20260327")
    parser.add_argument("--out_dir", type=str, default="outputs/data_audit")
    parser.add_argument("--train_sample_limit", type=int, default=None,
                        help="Per-camera train sample limit for image statistics.")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    issues = []

    if not args.skip_test:
        test_rows, test_issues = audit_test(Path(args.test_root))
        rows.extend(test_rows)
        issues.extend(test_issues)

    if not args.skip_train:
        train_rows, train_issues = audit_train(Path(args.train_root), args.train_sample_limit)
        rows.extend(train_rows)
        issues.extend(train_issues)

    summary = summarize(rows)
    write_csv(out_dir / "rail_data_audit_detail.csv", rows)
    write_csv(out_dir / "rail_data_audit_summary.csv", summary)

    report = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "test_root": args.test_root,
        "train_root": args.train_root,
        "num_rows": len(rows),
        "num_issues": len(issues),
        "issues": issues,
        "summary": summary,
    }
    with open(out_dir / "rail_data_audit_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Rows: {len(rows)}")
    print(f"Issues: {len(issues)}")
    print(f"Detail CSV: {out_dir / 'rail_data_audit_detail.csv'}")
    print(f"Summary CSV: {out_dir / 'rail_data_audit_summary.csv'}")
    print(f"Report JSON: {out_dir / 'rail_data_audit_report.json'}")
    if issues:
        print("\nFirst issues:")
        for issue in issues[:20]:
            print(f"- {issue}")


if __name__ == "__main__":
    main()
