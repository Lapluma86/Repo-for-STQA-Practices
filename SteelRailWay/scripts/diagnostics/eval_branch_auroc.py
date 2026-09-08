#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Cross-conditioned vs isolated branch AUROC diagnostics for Cam1/Cam4/Cam5."""

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
from datetime import datetime
from types import SimpleNamespace

import torch

from models.trd.encoder import ResNet50Encoder
from scripts.eval.eval_from_ckpt import (
    CROSS_SOURCES,
    ISOLATED_SOURCES,
    SCORE_SOURCES,
    ensure_assist_feature_means,
    evaluate_from_args,
    resolve_amp_dtype,
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


def load_ckpt_map(value: str | None) -> dict[int, str]:
    if not value:
        return {}
    path = Path(value)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    else:
        raw = json.loads(value)
    return {int(k): str(v) for k, v in raw.items() if v}


def resolve_default_cam4_peft_ckpt(peft_root: Path) -> str | None:
    candidates = sorted(
        peft_root.glob("cam4_p1_*/final/final_peft_cam4.pth"),
        key=lambda p: (p.parent.parent.stat().st_mtime, p.parent.parent.name),
        reverse=True,
    )
    if not candidates:
        return None
    return str(candidates[0])


def default_out_dir(depth_peft_ckpt: str | None) -> Path:
    if depth_peft_ckpt:
        peft_path = Path(depth_peft_ckpt)
        if peft_path.exists():
            return peft_path.parents[1] / "diagnostics" / "branch_isolation"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs/rail_peft") / f"branch_isolation_{timestamp}"


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate cross-conditioned vs isolated branch AUROC for Cam1/Cam4/Cam5."
    )
    parser.add_argument("--runs_root", type=str, default="outputs/rail_all")
    parser.add_argument("--views", type=int, nargs="+", default=[1, 4, 5])
    parser.add_argument("--ckpt_map", type=str, default=None,
                        help='可选：JSON 文件或 JSON 字符串，如 {"1": "path/to/best_cam1.pth"}')
    parser.add_argument("--depth_peft_ckpt", type=str, default=None,
                        help="可选：Cam4 final PEFT；不提供时自动从 outputs/rail_peft 查找最新结果")
    parser.add_argument("--train_root", type=str, default="data_20260327")
    parser.add_argument("--test_root", type=str, default="rail_mvtec_gt_test")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--assist_stats_batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--depth_norm", type=str, default="zscore",
                        choices=["zscore", "minmax", "log"])
    parser.add_argument("--precision", type=str, default="fp32",
                        choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--use_patch", action="store_true", default=True)
    parser.add_argument("--no_patch", action="store_false", dest="use_patch")
    parser.add_argument("--patch_size", type=int, default=900)
    parser.add_argument("--patch_stride", type=int, default=850)
    parser.add_argument("--preload", action="store_true")
    parser.add_argument("--preload_workers", type=int, default=8)
    parser.add_argument("--channels_last", action="store_true", default=True)
    parser.add_argument("--no_channels_last", action="store_false", dest="channels_last")
    parser.add_argument("--append_log", action="store_true", default=False)
    parser.add_argument("--assist_fill", type=str, default="train_mean",
                        choices=["train_mean", "zeros"])
    parser.add_argument("--train_sample_ratio", type=float, default=1.0,
                        help="用于 train-mean assist 特征统计的训练集采样比例")
    parser.add_argument("--train_sample_num", type=int, default=1200,
                        help="用于 train-mean assist 特征统计的训练集采样数量，默认贴近训练口径；<=0 表示使用全量")
    parser.add_argument("--sampling_mode", type=str, default="uniform_time",
                        choices=["uniform_time", "random"],
                        help="用于 train-mean assist 特征统计的训练集采样模式")
    parser.add_argument("--train_sample_seed", type=int, default=42,
                        help="用于 train-mean assist 特征统计的采样随机种子")
    return parser


def make_eval_args(args, ckpt: str, view_id: int, config_dir: Path,
                   depth_peft_ckpt: str | None, assist_stats_dir: Path):
    return SimpleNamespace(
        ckpt=str(ckpt),
        train_root=args.train_root,
        test_root=args.test_root,
        view_id=view_id,
        img_size=args.img_size,
        batch_size=args.batch_size,
        assist_stats_batch_size=args.assist_stats_batch_size,
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
        output_log=None,
        append_log=args.append_log,
        scores_csv=None,
        scores_dir=str(config_dir),
        result_json=str(config_dir / "result.json"),
        depth_peft_ckpt=str(depth_peft_ckpt or ""),
        score_source="fusion",
        assist_fill=args.assist_fill,
        assist_stats_dir=str(assist_stats_dir),
        train_sample_ratio=args.train_sample_ratio,
        train_sample_num=args.train_sample_num,
        sampling_mode=args.sampling_mode,
        train_sample_seed=args.train_sample_seed,
    )


def format_metric(value):
    return "" if value is None else f"{float(value):.8f}"


def metric_value(row: dict, key: str) -> float | None:
    text = row.get(key, "")
    return None if text == "" else float(text)


def branch_metric(result: dict, source: str):
    return result.get("auroc_by_source", {}).get(source)


def delta_key(source: str) -> str:
    if source == "fusion":
        return "delta_fusion_vs_baseline"
    if source in CROSS_SOURCES:
        return f"delta_{source}_cross_vs_baseline"
    return f"delta_{source}_vs_baseline"


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "view_id",
        "config_name",
        "ckpt",
        "depth_peft_ckpt",
        "assist_fill_mode",
        "assist_stats_dir",
        "auroc_rgb_cross",
        "auroc_depth_cross",
        "auroc_fusion",
        "auroc_rgb_isolated",
        "auroc_depth_isolated",
        "delta_rgb_cross_vs_baseline",
        "delta_depth_cross_vs_baseline",
        "delta_fusion_vs_baseline",
        "delta_rgb_isolated_vs_baseline",
        "delta_depth_isolated_vs_baseline",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def find_row(rows: list[dict], view_id: int, config_name: str) -> dict | None:
    for row in rows:
        if int(row["view_id"]) == int(view_id) and row["config_name"] == config_name:
            return row
    return None


def interpret_cam4_baseline(row: dict) -> str:
    depth_cross = metric_value(row, "auroc_depth_cross")
    depth_isolated = metric_value(row, "auroc_depth_isolated")
    if depth_cross is None or depth_isolated is None:
        return "Cam4 baseline: 缺少 depth_cross/depth_isolated，无法判断 Depth 本体与耦合影响。"
    if depth_isolated <= depth_cross + 0.05:
        return (
            f"Cam4 baseline: depth_cross={depth_cross:.4f}, depth_isolated={depth_isolated:.4f}，"
            "隔离后提升很小，说明 Depth 分支本体就偏弱，不只是被 crossmodal 污染。"
        )
    return (
        f"Cam4 baseline: depth_cross={depth_cross:.4f}, depth_isolated={depth_isolated:.4f}，"
        "隔离后明显变好，说明除了 Depth 漂移本身，还存在跨模态耦合放大问题。"
    )


def interpret_cam4_peft(baseline_row: dict, peft_row: dict) -> str:
    base_cross = metric_value(baseline_row, "auroc_depth_cross")
    peft_cross = metric_value(peft_row, "auroc_depth_cross")
    base_iso = metric_value(baseline_row, "auroc_depth_isolated")
    peft_iso = metric_value(peft_row, "auroc_depth_isolated")
    if None in {base_cross, peft_cross, base_iso, peft_iso}:
        return "Cam4 + PEFT: 缺少完整 depth 指标，无法判断收益主要落点。"
    delta_cross = peft_cross - base_cross
    delta_iso = peft_iso - base_iso
    if delta_iso >= delta_cross - 0.03:
        return (
            f"Cam4 + PEFT: depth_cross Δ={delta_cross:+.4f}, depth_isolated Δ={delta_iso:+.4f}，"
            "收益大体同步，说明 PEFT 主要在修 Depth 分支本体。"
        )
    return (
        f"Cam4 + PEFT: depth_cross Δ={delta_cross:+.4f}, depth_isolated Δ={delta_iso:+.4f}，"
        "cross-conditioned 提升更大，说明收益不只来自 Depth 本体，也包含耦合层面的恢复。"
    )


def interpret_cam5_effect(baseline_row: dict, peft_row: dict) -> str:
    base_cross_rgb = metric_value(baseline_row, "auroc_rgb_cross")
    base_cross_depth = metric_value(baseline_row, "auroc_depth_cross")
    base_iso_rgb = metric_value(baseline_row, "auroc_rgb_isolated")
    base_iso_depth = metric_value(baseline_row, "auroc_depth_isolated")
    peft_cross_rgb = metric_value(peft_row, "auroc_rgb_cross")
    peft_cross_depth = metric_value(peft_row, "auroc_depth_cross")
    peft_iso_rgb = metric_value(peft_row, "auroc_rgb_isolated")
    peft_iso_depth = metric_value(peft_row, "auroc_depth_isolated")
    if None in {
        base_cross_rgb, base_cross_depth, base_iso_rgb, base_iso_depth,
        peft_cross_rgb, peft_cross_depth, peft_iso_rgb, peft_iso_depth,
    }:
        return "Cam5 + Cam4 PEFT: 缺少完整指标，无法判断变化主要发生在 isolated 还是 cross-conditioned。"
    cross_rgb_delta = peft_cross_rgb - base_cross_rgb
    cross_depth_delta = peft_cross_depth - base_cross_depth
    iso_rgb_delta = peft_iso_rgb - base_iso_rgb
    iso_depth_delta = peft_iso_depth - base_iso_depth
    cross_sum = abs(cross_rgb_delta) + abs(cross_depth_delta)
    iso_sum = abs(iso_rgb_delta) + abs(iso_depth_delta)
    if iso_sum <= cross_sum * 0.5:
        return (
            f"Cam5 + Cam4 PEFT: cross Δ(rgb={cross_rgb_delta:+.4f}, depth={cross_depth_delta:+.4f}), "
            f"isolated Δ(rgb={iso_rgb_delta:+.4f}, depth={iso_depth_delta:+.4f})，"
            "isolated 变化更小，现象更像是 cross-conditioned / 融合排序重排。"
        )
    return (
        f"Cam5 + Cam4 PEFT: cross Δ(rgb={cross_rgb_delta:+.4f}, depth={cross_depth_delta:+.4f}), "
        f"isolated Δ(rgb={iso_rgb_delta:+.4f}, depth={iso_depth_delta:+.4f})，"
        "isolated 也明显变化，说明影响不只在耦合层，主分支表征本身也被带动。"
    )


def write_summary_txt(path: Path, args, depth_peft_ckpt: str | None, assist_stats_dir: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("Branch isolation diagnostics\n")
        f.write(f"created_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"runs_root: {args.runs_root}\n")
        f.write(f"train_root: {args.train_root}\n")
        f.write(f"test_root: {args.test_root}\n")
        f.write(f"views: {' '.join(map(str, args.views))}\n")
        f.write(f"device: {args.device}\n")
        f.write(f"precision: {args.precision}\n")
        f.write(f"batch_size: {args.batch_size}\n")
        f.write(f"assist_stats_batch_size: {args.assist_stats_batch_size}\n")
        f.write(f"num_workers: {args.num_workers}\n")
        f.write(f"use_patch: {args.use_patch}\n")
        f.write(f"assist_fill: {args.assist_fill}\n")
        f.write(f"train_sample_ratio: {args.train_sample_ratio}\n")
        f.write(f"train_sample_num: {args.train_sample_num}\n")
        f.write(f"sampling_mode: {args.sampling_mode}\n")
        f.write(f"train_sample_seed: {args.train_sample_seed}\n")
        f.write(f"assist_stats_dir: {assist_stats_dir}\n")
        f.write(f"depth_peft_ckpt: {depth_peft_ckpt or 'N/A'}\n\n")

        f.write("Cross-conditioned AUROC\n")
        for row in rows:
            f.write(
                f"Cam{row['view_id']} {row['config_name']}: "
                f"rgb_cross={row['auroc_rgb_cross']}, "
                f"depth_cross={row['auroc_depth_cross']}, "
                f"fusion={row['auroc_fusion']}, "
                f"d_rgb_cross={row['delta_rgb_cross_vs_baseline']}, "
                f"d_depth_cross={row['delta_depth_cross_vs_baseline']}, "
                f"d_fusion={row['delta_fusion_vs_baseline']}\n"
            )

        f.write("\nIsolated AUROC\n")
        for row in rows:
            f.write(
                f"Cam{row['view_id']} {row['config_name']}: "
                f"rgb_isolated={row['auroc_rgb_isolated']}, "
                f"depth_isolated={row['auroc_depth_isolated']}, "
                f"d_rgb_isolated={row['delta_rgb_isolated_vs_baseline']}, "
                f"d_depth_isolated={row['delta_depth_isolated_vs_baseline']}\n"
            )

        f.write("\nInterpretation\n")
        cam4_baseline = find_row(rows, 4, "baseline")
        cam4_peft = find_row(rows, 4, "with_cam4peft")
        cam5_baseline = find_row(rows, 5, "baseline")
        cam5_peft = find_row(rows, 5, "with_cam4peft")
        if cam4_baseline:
            f.write(f"- {interpret_cam4_baseline(cam4_baseline)}\n")
        if cam4_baseline and cam4_peft:
            f.write(f"- {interpret_cam4_peft(cam4_baseline, cam4_peft)}\n")
        if cam5_baseline and cam5_peft:
            f.write(f"- {interpret_cam5_effect(cam5_baseline, cam5_peft)}\n")


def precompute_assist_stats(args, view_id: int, assist_stats_dir: Path) -> None:
    print("\n" + "-" * 72)
    print(f"Preparing train-mean assist features for Cam{view_id}")
    print("-" * 72)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    amp_dtype = resolve_amp_dtype(args.precision, device)
    teacher_rgb = ResNet50Encoder(pretrained=True).to(device).eval()
    teacher_depth = ResNet50Encoder(pretrained=True).to(device).eval()
    for param in teacher_rgb.parameters():
        param.requires_grad_(False)
    for param in teacher_depth.parameters():
        param.requires_grad_(False)

    if args.channels_last and device.type == "cuda":
        teacher_rgb = teacher_rgb.to(memory_format=torch.channels_last)
        teacher_depth = teacher_depth.to(memory_format=torch.channels_last)

    stats_args = SimpleNamespace(
        train_root=args.train_root,
        test_root=args.test_root,
        view_id=view_id,
        img_size=args.img_size,
        depth_norm=args.depth_norm,
        preload=args.preload,
        preload_workers=args.preload_workers,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        assist_stats_batch_size=args.assist_stats_batch_size,
        channels_last=args.channels_last,
        assist_fill=args.assist_fill,
        assist_stats_dir=str(assist_stats_dir),
        train_sample_ratio=args.train_sample_ratio,
        train_sample_num=args.train_sample_num,
        sampling_mode=args.sampling_mode,
        train_sample_seed=args.train_sample_seed,
    )
    ensure_assist_feature_means(
        stats_args,
        device,
        amp_dtype,
        teacher_rgb,
        teacher_depth,
    )

    del teacher_rgb, teacher_depth
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    args = build_parser().parse_args()

    runs_root = Path(args.runs_root)
    ckpt_map = load_ckpt_map(args.ckpt_map)
    depth_peft_ckpt = args.depth_peft_ckpt or resolve_default_cam4_peft_ckpt(Path("outputs/rail_peft"))
    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir(depth_peft_ckpt)
    out_dir.mkdir(parents=True, exist_ok=True)
    assist_stats_dir = out_dir / "reference_stats"
    assist_stats_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output dir: {out_dir}")
    print(f"Assist stats dir: {assist_stats_dir}")
    if depth_peft_ckpt:
        print(f"Depth PEFT: {depth_peft_ckpt}")
    else:
        print("Depth PEFT: N/A (baseline only)")

    for view_id in args.views:
        precompute_assist_stats(args, view_id, assist_stats_dir)

    configs = [("baseline", None)]
    if depth_peft_ckpt:
        configs.append(("with_cam4peft", depth_peft_ckpt))

    rows = []
    baseline_by_view = {}
    for view_id in args.views:
        ckpt = ckpt_map.get(view_id)
        if not ckpt:
            found = find_latest_ckpt(runs_root, view_id)
            if found is None:
                raise FileNotFoundError(f"Checkpoint not found for Cam{view_id} under {runs_root}")
            ckpt = str(found)

        for config_name, peft_path in configs:
            print("\n" + "=" * 72)
            print(f"Evaluating Cam{view_id} / {config_name}")
            print("=" * 72)
            config_dir = out_dir / f"cam{view_id}_{config_name}"
            config_dir.mkdir(parents=True, exist_ok=True)
            eval_args = make_eval_args(args, ckpt, view_id, config_dir, peft_path, assist_stats_dir)
            result = evaluate_from_args(eval_args)

            row = {
                "view_id": int(view_id),
                "config_name": config_name,
                "ckpt": str(ckpt),
                "depth_peft_ckpt": str(peft_path or ""),
                "assist_fill_mode": args.assist_fill,
                "assist_stats_dir": str(assist_stats_dir),
                "auroc_rgb_cross": format_metric(branch_metric(result, "rgb")),
                "auroc_depth_cross": format_metric(branch_metric(result, "depth")),
                "auroc_fusion": format_metric(branch_metric(result, "fusion")),
                "auroc_rgb_isolated": format_metric(branch_metric(result, "rgb_isolated")),
                "auroc_depth_isolated": format_metric(branch_metric(result, "depth_isolated")),
                "delta_rgb_cross_vs_baseline": "",
                "delta_depth_cross_vs_baseline": "",
                "delta_fusion_vs_baseline": "",
                "delta_rgb_isolated_vs_baseline": "",
                "delta_depth_isolated_vs_baseline": "",
            }
            rows.append(row)
            if config_name == "baseline":
                baseline_by_view[view_id] = {
                    source: branch_metric(result, source)
                    for source in SCORE_SOURCES
                }

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    for row in rows:
        baseline = baseline_by_view.get(row["view_id"], {})
        for source in SCORE_SOURCES:
            if source == "fusion":
                metric_key = "auroc_fusion"
            elif source in CROSS_SOURCES:
                metric_key = f"auroc_{source}_cross"
            else:
                metric_key = f"auroc_{source}"
            metric_text = row[metric_key]
            if not metric_text:
                row[delta_key(source)] = ""
                continue
            metric_value_current = float(metric_text)
            baseline_value = baseline.get(source)
            if baseline_value is None:
                row[delta_key(source)] = ""
            else:
                row[delta_key(source)] = f"{metric_value_current - float(baseline_value):.8f}"

    summary_csv = out_dir / "summary.csv"
    summary_txt = out_dir / "summary.txt"
    write_summary_csv(summary_csv, rows)
    write_summary_txt(summary_txt, args, depth_peft_ckpt, assist_stats_dir, rows)

    print("\n" + "=" * 72)
    print("Branch-isolation diagnostics finished")
    print("=" * 72)
    print(f"Summary CSV: {summary_csv}")
    print(f"Summary TXT: {summary_txt}")


if __name__ == "__main__":
    main()
