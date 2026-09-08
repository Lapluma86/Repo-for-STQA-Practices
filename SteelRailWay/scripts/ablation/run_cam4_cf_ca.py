#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run Cam4 post-hoc CF/CA module-path ablations from an existing checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace


_PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

MODULE_ABLATION_MODES = ("full", "no_cf", "no_ca", "no_cf_ca")


DEFAULT_CKPT = (
    "outputs/rail_all/Cam4/20260501_014226_cam4_bs32_lr0.005_img512_ratio1.0/"
    "best_cam4.pth"
)
DEFAULT_PEFT_CKPT = (
    "outputs/rail_peft/cam4_p1_20260501_225618/final/final_peft_cam4.pth"
)

SCHEME_LABELS = {
    "full": "完整模型",
    "no_cf": "去除 CF",
    "no_ca": "去除 CA",
    "no_cf_ca": "去除 CF+CA",
}

SCHEME_NOTES = {
    "full": "CF+CA",
    "no_cf": "禁用瓶颈跨模态 filter 融合",
    "no_ca": "禁用解码阶段辅助 skip 融合",
    "no_cf_ca": "两条跨模态路径均禁用",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Cam4 CF/CA post-hoc ablations and write a paper-ready summary."
    )
    parser.add_argument("--ckpt", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--depth_peft_ckpt", type=str, default=None,
                        help="可选：DepthAffinePEFT checkpoint；提供时对同一 student ckpt 做 PEFT+CF/CA 消融")
    parser.add_argument("--train_root", type=str, default="data_20260327")
    parser.add_argument("--test_root", type=str, default="rail_mvtec_gt_test")
    parser.add_argument("--out_root", type=str, default="outputs/rail_ablation/cam4_cf_ca")
    parser.add_argument("--modes", nargs="+", default=list(MODULE_ABLATION_MODES),
                        choices=MODULE_ABLATION_MODES)
    parser.add_argument("--view_id", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--depth_norm", type=str, default="zscore",
                        choices=["zscore", "minmax", "log"])
    parser.add_argument("--precision", type=str, default="fp32",
                        choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--fusion_rule", type=str, default="sum",
                        choices=["sum", "max_norm"])
    parser.add_argument("--patch_size", type=int, default=900)
    parser.add_argument("--patch_stride", type=int, default=850)
    parser.add_argument("--assist_fill", type=str, default="zeros",
                        choices=["zeros", "train_mean"],
                        help="Use zeros by default because isolated branches are not used in Table 6.6.2.")
    parser.add_argument("--assist_stats_dir", type=str, default=None)
    parser.add_argument("--preload", action="store_true")
    parser.add_argument("--preload_workers", type=int, default=8)
    parser.add_argument("--channels_last", action="store_true", default=True)
    parser.add_argument("--no_channels_last", action="store_false", dest="channels_last")
    parser.add_argument("--strict_full_baseline", action="store_true",
                        help="Exit non-zero if full mode does not reproduce the known Cam4 baseline.")
    parser.add_argument("--baseline_tolerance", type=float, default=1e-4)
    parser.add_argument("--expected_full_rgb", type=float, default=None,
                        help="可选：full 模式期望的 RGB AUROC；提供后用于复现检查")
    parser.add_argument("--expected_full_depth", type=float, default=None,
                        help="可选：full 模式期望的 Depth AUROC；提供后用于复现检查")
    parser.add_argument("--expected_full_fusion", type=float, default=None,
                        help="可选：full 模式期望的 Fusion AUROC；提供后用于复现检查")
    return parser


def make_eval_args(args: argparse.Namespace, mode: str, mode_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        ckpt=args.ckpt,
        train_root=args.train_root,
        test_root=args.test_root,
        view_id=args.view_id,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        depth_norm=args.depth_norm,
        use_patch=True,
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        preload=args.preload,
        preload_workers=args.preload_workers,
        precision=args.precision,
        channels_last=args.channels_last,
        output_log=None,
        append_log=False,
        scores_csv=None,
        result_json=str(mode_dir / "result.json"),
        depth_peft_ckpt=args.depth_peft_ckpt,
        module_ablation=mode,
        score_source="fusion",
        fusion_rule=args.fusion_rule,
        scores_dir=str(mode_dir),
        assist_fill=args.assist_fill,
        assist_stats_dir=args.assist_stats_dir,
        train_sample_ratio=1.0,
        train_sample_num=None,
        sampling_mode="uniform_time",
        train_sample_seed=42,
    )


def metric(result: dict, source: str) -> float | None:
    value = result.get("auroc_by_source", {}).get(source)
    return None if value is None else float(value)


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.8f}"


def write_summary(path: Path, rows: list[dict]) -> None:
    fields = [
        "module_ablation",
        "scheme",
        "auroc_rgb",
        "auroc_depth",
        "auroc_fusion",
        "delta_fusion_vs_full",
        "num_images",
        "num_abnormal",
        "num_normal",
        "note",
        "result_json",
        "depth_peft_ckpt",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def resolve_expected_full(args: argparse.Namespace) -> dict[str, float] | None:
    custom = {
        "auroc_rgb": args.expected_full_rgb,
        "auroc_depth": args.expected_full_depth,
        "auroc_fusion": args.expected_full_fusion,
    }
    if all(value is not None for value in custom.values()):
        return {k: float(v) for k, v in custom.items()}
    if any(value is not None for value in custom.values()):
        raise ValueError("expected_full_rgb/depth/fusion 必须同时提供，或全部不提供")
    if args.depth_peft_ckpt:
        return None
    return {
        "auroc_rgb": 0.7000,
        "auroc_depth": 0.3625,
        "auroc_fusion": 0.3500,
    }


def check_full_baseline(rows: list[dict], expected: dict[str, float] | None, tolerance: float) -> list[str]:
    if expected is None:
        return []
    full = next((row for row in rows if row["module_ablation"] == "full"), None)
    if not full:
        return ["full mode was not evaluated"]
    warnings = []
    for key, expected_value in expected.items():
        value = float(full[key])
        if abs(value - expected_value) > tolerance:
            warnings.append(f"{key}: got {value:.8f}, expected {expected_value:.8f}")
    return warnings


def main() -> None:
    args = build_parser().parse_args()
    from scripts.eval.eval_from_ckpt import evaluate_from_args

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    result_by_mode = {}
    for mode in args.modes:
        mode_dir = out_root / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        print("\n" + "=" * 72)
        print(f"Cam4 CF/CA ablation: {mode}")
        print("=" * 72)
        result = evaluate_from_args(make_eval_args(args, mode, mode_dir))
        result_by_mode[mode] = result

    full_fusion = metric(result_by_mode.get("full", {}), "fusion")
    rows = []
    for mode in args.modes:
        result = result_by_mode[mode]
        fusion = metric(result, "fusion")
        rows.append({
            "module_ablation": mode,
            "scheme": SCHEME_LABELS[mode],
            "auroc_rgb": fmt(metric(result, "rgb")),
            "auroc_depth": fmt(metric(result, "depth")),
            "auroc_fusion": fmt(fusion),
            "delta_fusion_vs_full": fmt(None if full_fusion is None or fusion is None else fusion - full_fusion),
            "num_images": result.get("num_images", ""),
            "num_abnormal": result.get("num_abnormal", ""),
            "num_normal": result.get("num_normal", ""),
            "note": SCHEME_NOTES[mode],
            "result_json": str(out_root / mode / "result.json"),
            "depth_peft_ckpt": result.get("depth_peft_ckpt", ""),
        })

    summary_csv = out_root / "summary.csv"
    write_summary(summary_csv, rows)
    with open(out_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"\nSummary CSV: {summary_csv}")

    expected_full = resolve_expected_full(args)
    warnings = check_full_baseline(rows, expected_full, args.baseline_tolerance)
    if warnings:
        print("Baseline reproduction warnings:")
        for warning in warnings:
            print(f"  - {warning}")
        if args.strict_full_baseline:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
