#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train and evaluate Cam4 depth-normalization ablations."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


_PROJ_ROOT = Path(__file__).resolve().parents[2]

NORM_NOTES = {
    "zscore": "当前主方案",
    "minmax": "对极值敏感",
    "log": "压缩大深度值动态范围",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Cam4 zscore/minmax/log depth normalization retraining ablations."
    )
    parser.add_argument("--train_root", type=str, default="/data1/Leaddo_data/20260327-resize512")
    parser.add_argument("--test_root", type=str, default="/home/root123/LF/WYM/SteelRailWay/rail_mvtec_gt_test")
    parser.add_argument("--out_root", type=str, default="outputs/rail_ablation/depth_norm/Cam4")
    parser.add_argument("--norms", nargs="+", default=["zscore", "minmax", "log"],
                        choices=["zscore", "minmax", "log"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--preload_workers", type=int, default=32)
    parser.add_argument("--precision", type=str, default="bf16",
                        choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--train_sample_num", type=int, default=1200)
    parser.add_argument("--sampling_mode", type=str, default="uniform_time",
                        choices=["uniform_time", "random"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random_seed_sample", type=int, default=42)
    parser.add_argument("--test_patch_size", type=int, default=900)
    parser.add_argument("--test_patch_stride", type=int, default=850)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--eval_num_workers", type=int, default=0)
    parser.add_argument("--eval_precision", type=str, default="fp32",
                        choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip training and evaluate latest checkpoint under each norm output dir.")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip training if a best_cam4.pth already exists under the norm output dir.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without executing them.")
    return parser


def run_command(cmd: list[str], dry_run: bool) -> None:
    print("\n" + " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=_PROJ_ROOT, check=True)


def latest_ckpt(norm_dir: Path) -> Path | None:
    candidates = sorted(
        norm_dir.glob("*cam4_*/best_cam4.pth"),
        key=lambda p: (p.parent.stat().st_mtime, p.parent.name),
        reverse=True,
    )
    if candidates:
        return candidates[0]
    direct = norm_dir / "best_cam4.pth"
    return direct if direct.exists() else None


def train_command(args: argparse.Namespace, norm: str, norm_dir: Path) -> list[str]:
    return [
        sys.executable,
        "train/train_trd_rail.py",
        "--train_root", args.train_root,
        "--test_root", args.test_root,
        "--view_id", "4",
        "--img_size", str(args.img_size),
        "--depth_norm", norm,
        "--batch_size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--num_workers", str(args.num_workers),
        "--preload",
        "--preload_workers", str(args.preload_workers),
        "--precision", args.precision,
        "--channels_last",
        "--train_sample_num", str(args.train_sample_num),
        "--sampling_mode", args.sampling_mode,
        "--seed", str(args.seed),
        "--random_seed_sample", str(args.random_seed_sample),
        "--test_patch_size", str(args.test_patch_size),
        "--test_patch_stride", str(args.test_patch_stride),
        "--device", args.device,
        "--save_dir", str(norm_dir),
        "--no_compile",
    ]


def eval_command(args: argparse.Namespace, norm: str, ckpt: Path, eval_dir: Path) -> list[str]:
    return [
        sys.executable,
        "scripts/eval/eval_from_ckpt.py",
        "--ckpt", str(ckpt),
        "--train_root", args.train_root,
        "--test_root", args.test_root,
        "--view_id", "4",
        "--img_size", str(args.img_size),
        "--depth_norm", norm,
        "--batch_size", str(args.eval_batch_size),
        "--num_workers", str(args.eval_num_workers),
        "--device", args.device,
        "--precision", args.eval_precision,
        "--use_patch",
        "--patch_size", str(args.test_patch_size),
        "--patch_stride", str(args.test_patch_stride),
        "--score_source", "fusion",
        "--module_ablation", "full",
        "--assist_fill", "zeros",
        "--scores_dir", str(eval_dir),
        "--result_json", str(eval_dir / "result.json"),
    ]


def fmt(value) -> str:
    return "" if value is None else f"{float(value):.8f}"


def write_summary(out_root: Path, norms: list[str]) -> Path:
    rows = []
    for norm in norms:
        result_path = out_root / norm / "eval" / "result.json"
        if not result_path.exists():
            rows.append({
                "depth_norm": norm,
                "auroc_rgb": "",
                "auroc_depth": "",
                "auroc_fusion": "",
                "best_epoch": "",
                "best_val_loss": "",
                "num_images": "",
                "num_abnormal": "",
                "num_normal": "",
                "note": NORM_NOTES[norm],
                "result_json": str(result_path),
            })
            continue
        with open(result_path, "r", encoding="utf-8") as f:
            result = json.load(f)
        auroc = result.get("auroc_by_source", {})
        rows.append({
            "depth_norm": norm,
            "auroc_rgb": fmt(auroc.get("rgb")),
            "auroc_depth": fmt(auroc.get("depth")),
            "auroc_fusion": fmt(auroc.get("fusion")),
            "best_epoch": result.get("best_epoch", ""),
            "best_val_loss": result.get("best_val_loss", ""),
            "num_images": result.get("num_images", ""),
            "num_abnormal": result.get("num_abnormal", ""),
            "num_normal": result.get("num_normal", ""),
            "note": NORM_NOTES[norm],
            "result_json": str(result_path),
        })

    summary_csv = out_root / "summary.csv"
    fields = [
        "depth_norm",
        "auroc_rgb",
        "auroc_depth",
        "auroc_fusion",
        "best_epoch",
        "best_val_loss",
        "num_images",
        "num_abnormal",
        "num_normal",
        "note",
        "result_json",
    ]
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return summary_csv


def main() -> None:
    args = build_parser().parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if not args.eval_only and not args.dry_run and not Path(args.train_root).exists():
        raise FileNotFoundError(
            f"train_root not found: {args.train_root}. "
            "Run this on the training server or pass a restored local training root."
        )

    for norm in args.norms:
        norm_dir = out_root / norm
        norm_dir.mkdir(parents=True, exist_ok=True)
        ckpt = latest_ckpt(norm_dir)
        if not args.eval_only and not (args.skip_existing and ckpt):
            run_command(train_command(args, norm, norm_dir), args.dry_run)
            ckpt = latest_ckpt(norm_dir)

        if args.dry_run:
            ckpt = ckpt or norm_dir / "<timestamp>_cam4_.../best_cam4.pth"
        if ckpt is None:
            raise FileNotFoundError(f"No best_cam4.pth found under {norm_dir}")

        eval_dir = norm_dir / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        run_command(eval_command(args, norm, ckpt, eval_dir), args.dry_run)

    if not args.dry_run:
        summary_csv = write_summary(out_root, args.norms)
        print(f"\nSummary CSV: {summary_csv}")


if __name__ == "__main__":
    main()

