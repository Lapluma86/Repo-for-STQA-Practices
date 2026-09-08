#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量补评估 outputs/rail_all 下的单视角模型。

默认行为：
    - 自动查找 outputs/rail_all/CamN/*camN*/best_camN.pth
      也兼容旧的 outputs/rail_all/*camN*/best_camN.pth
    - 对 cam1-8 顺序评估
    - 把每个 cam 的最终结果追加回对应 run 的 training.log
    - 在 outputs/rail_all/_summaries 下生成 eval_summary_*.csv / eval_summary_*.txt

注意：当前 rail_mvtec_gt_test 只有 cam1-6，cam7/cam8 会被记录为 N/A。
"""

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
import gc
import json
import traceback
from datetime import datetime
from types import SimpleNamespace

import torch

from rail_cad.registry import build_active_depth_peft_map, load_registry
from scripts.eval.eval_from_ckpt import (
    SCORE_SOURCES,
    append_eval_log,
    default_scores_csv_path,
    evaluate_from_args,
)


def find_latest_ckpt(runs_root: Path, view_id: int) -> Path | None:
    patterns = [
        runs_root / f"Cam{view_id}" / f"*cam{view_id}_*" / f"best_cam{view_id}.pth",
        runs_root / f"*cam{view_id}_*" / f"best_cam{view_id}.pth",
    ]
    candidates = []
    seen = set()
    for pattern in patterns:
        for path in runs_root.glob(str(pattern.relative_to(runs_root))):
            resolved = path.resolve()
            if resolved not in seen:
                candidates.append(path)
                seen.add(resolved)
    if not candidates:
        return None
    candidates.sort(key=lambda p: (p.parent.stat().st_mtime, p.parent.name), reverse=True)
    return candidates[0]


def base_result(args, view_id, ckpt=None, run_dir=None, log_file=None):
    return {
        "evaluated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "view_id": int(view_id),
        "run_dir": str(run_dir) if run_dir else "",
        "ckpt": str(ckpt) if ckpt else "",
        "depth_peft_ckpt": "",
        "score_source_selected": str(getattr(args, "score_source", "fusion")),
        "train_root": str(args.train_root),
        "test_root": str(args.test_root),
        "status": "skipped",
        "reason": "",
        "best_epoch": "N/A",
        "best_val_loss": None,
        "auroc": None,
        "num_patches": 0,
        "num_images": 0,
        "num_abnormal": 0,
        "num_normal": 0,
        "score_min": None,
        "score_max": None,
        "scores_csv": "",
        "output_log": str(log_file) if log_file else "",
        "log_file": str(log_file) if log_file else "",
    }


def format_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def append_summary_text(path: Path, result: dict):
    auroc = "N/A" if result.get("auroc") is None else f"{result['auroc']:.4f}"
    best_val = result.get("best_val_loss")
    best_val_text = "N/A" if best_val is None else f"{float(best_val):.4f}"
    line = (
        f"Cam{result['view_id']}: status={result['status']}, "
        f"score_source={result.get('score_source_selected', 'fusion')}, "
        f"AUROC={auroc}, best_epoch={result.get('best_epoch', 'N/A')}, "
        f"best_val_loss={best_val_text}, images={result['num_images']}, "
        f"abnormal={result['num_abnormal']}, normal={result['num_normal']}"
    )
    if result.get("reason"):
        line += f", reason={result['reason']}"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def write_csv_row(writer, result):
    writer.writerow({
        "view_id": result.get("view_id"),
        "status": result.get("status"),
        "reason": result.get("reason"),
        "score_source": result.get("score_source_selected"),
        "auroc": format_value(result.get("auroc")),
        "best_epoch": result.get("best_epoch"),
        "best_val_loss": format_value(result.get("best_val_loss")),
        "num_patches": result.get("num_patches"),
        "num_images": result.get("num_images"),
        "num_abnormal": result.get("num_abnormal"),
        "num_normal": result.get("num_normal"),
        "score_min": format_value(result.get("score_min")),
        "score_max": format_value(result.get("score_max")),
        "run_dir": result.get("run_dir"),
        "ckpt": result.get("ckpt"),
        "depth_peft_ckpt": result.get("depth_peft_ckpt"),
        "scores_csv": result.get("scores_csv"),
        "log_file": result.get("log_file") or result.get("output_log"),
    })


def load_depth_peft_map(value: str | None) -> dict[int, str]:
    if not value:
        return {}
    path = Path(value)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    else:
        raw = json.loads(value)
    return {int(k): str(v) for k, v in raw.items() if v}


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate all rail single-view checkpoints.")
    parser.add_argument("--runs_root", type=str, default="outputs/rail_all")
    parser.add_argument("--summary_dir", type=str, default=None,
                        help="默认写到 <runs_root>/_summaries")
    parser.add_argument("--train_root", type=str,
                        default=os.environ.get("TRAIN_ROOT", "/data1/Leaddo_data/20260327-resize512"))
    parser.add_argument("--test_root", type=str,
                        default=os.environ.get("TEST_ROOT", "/home/root123/LF/WYM/SteelRailWay/rail_mvtec_gt_test"))
    parser.add_argument("--views", type=int, nargs="+", default=list(range(1, 9)))
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--depth_norm", type=str, default="zscore",
                        choices=["zscore", "minmax", "log"])
    parser.add_argument("--precision", type=str, default="bf16",
                        choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--use_patch", action="store_true", default=True)
    parser.add_argument("--no_patch", action="store_false", dest="use_patch")
    parser.add_argument("--patch_size", type=int, default=900)
    parser.add_argument("--patch_stride", type=int, default=850)
    parser.add_argument("--preload", action="store_true")
    parser.add_argument("--preload_workers", type=int, default=8)
    parser.add_argument("--channels_last", action="store_true", default=True)
    parser.add_argument("--no_channels_last", action="store_false", dest="channels_last")
    parser.add_argument("--append_log", action="store_true", default=True)
    parser.add_argument("--no_append_log", action="store_false", dest="append_log")
    parser.add_argument("--depth_peft_map", type=str, default=None,
                        help='可选：JSON 文件或 JSON 字符串，如 {"4": "path/to/final_peft_cam4.pth"}')
    parser.add_argument("--cad_registry", type=str, default=None,
                        help="可选：CAD registry.json 或其根目录；提供时按 registry 导出视角路由")
    parser.add_argument("--score_source", type=str, default="fusion",
                        choices=SCORE_SOURCES,
                        help="顶层 AUROC / scores_csv 使用的分支来源，默认 fusion")
    parser.add_argument("--assist_fill", type=str, default="train_mean",
                        choices=["train_mean", "zeros"],
                        help="isolated 分支的辅助特征替代策略，默认 train_mean")
    parser.add_argument("--assist_stats_dir", type=str, default=None,
                        help="可选：训练域辅助特征均值缓存目录")
    parser.add_argument("--assist_stats_batch_size", type=int, default=16)
    parser.add_argument("--train_sample_ratio", type=float, default=1.0,
                        help="仅用于 isolated train-mean 统计的训练集采样比例")
    parser.add_argument("--train_sample_num", type=int, default=None,
                        help="仅用于 isolated train-mean 统计的训练集采样数量；<=0 表示使用全量")
    parser.add_argument("--sampling_mode", type=str, default="uniform_time",
                        choices=["uniform_time", "random"],
                        help="仅用于 isolated train-mean 统计的训练集采样模式")
    parser.add_argument("--train_sample_seed", type=int, default=42,
                        help="仅用于 isolated train-mean 统计的训练集采样随机种子")
    return parser


def main():
    args = build_parser().parse_args()
    runs_root = Path(args.runs_root)
    if args.cad_registry:
        registry = load_registry(args.cad_registry)
        depth_peft_map = build_active_depth_peft_map(registry)
    else:
        depth_peft_map = load_depth_peft_map(args.depth_peft_map)
    runs_root.mkdir(parents=True, exist_ok=True)
    summary_root = Path(args.summary_dir) if args.summary_dir else runs_root / "_summaries"
    summary_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_csv = summary_root / f"eval_summary_{timestamp}.csv"
    summary_txt = summary_root / f"eval_summary_{timestamp}.txt"

    fields = [
        "view_id", "status", "reason", "score_source", "auroc", "best_epoch", "best_val_loss",
        "num_patches", "num_images", "num_abnormal", "num_normal",
        "score_min", "score_max", "run_dir", "ckpt", "depth_peft_ckpt",
        "scores_csv", "log_file",
    ]

    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(f"Evaluation started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Runs root: {runs_root}\n")
        f.write(f"Train root: {args.train_root}\n")
        f.write(f"Test root: {args.test_root}\n")
        f.write(f"Views: {' '.join(map(str, args.views))}\n\n")
        f.write(f"Score source: {args.score_source}\n\n")
        if args.cad_registry:
            f.write(f"CAD registry: {args.cad_registry}\n\n")
        if depth_peft_map:
            f.write(f"Depth PEFT map: {depth_peft_map}\n\n")

    with open(summary_csv, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=fields)
        writer.writeheader()

        for view_id in args.views:
            print("\n" + "=" * 72)
            print(f"Evaluating Cam{view_id}")
            print("=" * 72)

            ckpt = find_latest_ckpt(runs_root, view_id)
            if ckpt is None:
                result = base_result(args, view_id)
                result["reason"] = "checkpoint not found"
                print(f"Skipped Cam{view_id}: checkpoint not found")
                append_summary_text(summary_txt, result)
                write_csv_row(writer, result)
                f_csv.flush()
                continue

            run_dir = ckpt.parent
            log_file = run_dir / "training.log"
            scores_csv = Path(default_scores_csv_path(ckpt, args.score_source))
            result_json = run_dir / f"eval_cam{view_id}_result.json"
            depth_peft_ckpt = "" if args.cad_registry else depth_peft_map.get(view_id, "")

            eval_args = SimpleNamespace(
                ckpt=str(ckpt),
                train_root=args.train_root,
                test_root=args.test_root,
                view_id=view_id,
                img_size=args.img_size,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=args.device,
                depth_norm=args.depth_norm,
                use_patch=args.use_patch,
                patch_size=args.patch_size,
                patch_stride=args.patch_stride,
                preload=args.preload,
                preload_workers=args.preload_workers,
                precision=args.precision,
                channels_last=args.channels_last,
                output_log=str(log_file),
                append_log=args.append_log,
                scores_csv=str(scores_csv),
                result_json=str(result_json),
                depth_peft_ckpt=depth_peft_ckpt,
                cad_registry=args.cad_registry,
                score_source=args.score_source,
                scores_dir=None,
                assist_fill=args.assist_fill,
                assist_stats_dir=args.assist_stats_dir,
                assist_stats_batch_size=args.assist_stats_batch_size,
                train_sample_ratio=args.train_sample_ratio,
                train_sample_num=args.train_sample_num,
                sampling_mode=args.sampling_mode,
                train_sample_seed=args.train_sample_seed,
            )

            try:
                result = evaluate_from_args(eval_args)
                result["run_dir"] = str(run_dir)
                result["log_file"] = str(log_file)
            except Exception as exc:
                result = base_result(args, view_id, ckpt=ckpt, run_dir=run_dir, log_file=log_file)
                result["status"] = "failed"
                result["reason"] = f"{type(exc).__name__}: {exc}"
                print(f"Failed Cam{view_id}: {result['reason']}")
                traceback.print_exc()
                if args.append_log:
                    append_eval_log(str(log_file), result)

            append_summary_text(summary_txt, result)
            write_csv_row(writer, result)
            f_csv.flush()

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\n" + "=" * 72)
    print("Batch evaluation finished")
    print("=" * 72)
    print(f"Summary CSV: {summary_csv}")
    print(f"Summary TXT: {summary_txt}")


if __name__ == "__main__":
    main()
