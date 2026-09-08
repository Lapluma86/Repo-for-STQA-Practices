#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Re-run Cam4 CF/CA ablations under multiple fusion rules."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

_PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from scripts.eval.eval_from_ckpt import evaluate_from_args


DEFAULT_BASELINE_CKPT = (
    "outputs/rail_all/Cam4/20260501_014226_cam4_bs32_lr0.005_img512_ratio1.0/best_cam4.pth"
)
DEFAULT_PEFT_CKPT = "outputs/rail_peft/cam4_p1_20260501_225618/final/final_peft_cam4.pth"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Cam4 CF/CA ablations with fusion_rule=sum|max_norm."
    )
    parser.add_argument("--ckpt", type=str, default=DEFAULT_BASELINE_CKPT)
    parser.add_argument("--depth_peft_ckpt", type=str, default=DEFAULT_PEFT_CKPT)
    parser.add_argument("--train_root", type=str, default="data_20260327")
    parser.add_argument("--test_root", type=str, default="rail_mvtec_gt_test")
    parser.add_argument("--view_id", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--precision", type=str, default="fp32", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--depth_norm", type=str, default="zscore", choices=["zscore", "minmax", "log"])
    parser.add_argument("--patch_size", type=int, default=900)
    parser.add_argument("--patch_stride", type=int, default=850)
    parser.add_argument("--assist_fill", type=str, default="zeros", choices=["zeros", "train_mean"])
    parser.add_argument("--out_dir", type=str, default="outputs/rail_ablation/cam4_cf_ca_peft_fusion_rules")
    return parser


def make_args(ns: argparse.Namespace, mode: str, fusion_rule: str, out_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        ckpt=ns.ckpt,
        train_root=ns.train_root,
        test_root=ns.test_root,
        view_id=ns.view_id,
        img_size=ns.img_size,
        batch_size=ns.batch_size,
        num_workers=ns.num_workers,
        device=ns.device,
        depth_norm=ns.depth_norm,
        use_patch=True,
        patch_size=ns.patch_size,
        patch_stride=ns.patch_stride,
        preload=False,
        preload_workers=0,
        precision=ns.precision,
        channels_last=True,
        output_log=None,
        append_log=False,
        scores_csv=None,
        result_json=str(out_dir / "result.json"),
        depth_peft_ckpt=ns.depth_peft_ckpt,
        module_ablation=mode,
        score_source="fusion",
        fusion_rule=fusion_rule,
        scores_dir=str(out_dir),
        assist_fill=ns.assist_fill,
        assist_stats_dir=None,
        train_sample_ratio=1.0,
        train_sample_num=None,
        sampling_mode="uniform_time",
        train_sample_seed=42,
    )


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    full_sum = None
    for fusion_rule in ["sum", "max_norm"]:
        for mode in ["full", "no_cf", "no_ca", "no_cf_ca"]:
            mode_dir = out_dir / fusion_rule / mode
            mode_dir.mkdir(parents=True, exist_ok=True)
            result = evaluate_from_args(make_args(args, mode, fusion_rule, mode_dir))
            fusion_auroc = float(result["auroc_by_source"]["fusion"])
            if fusion_rule == "sum" and mode == "full":
                full_sum = fusion_auroc
            rows.append({
                "module_ablation": mode,
                "fusion_rule": fusion_rule,
                "fusion_auroc": f"{fusion_auroc:.8f}",
                "delta_vs_full_sum": "" if full_sum is None else f"{(fusion_auroc - full_sum):.8f}",
                "result_json": str(mode_dir / "result.json"),
            })

    out_csv = out_dir / "summary.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "rows": rows,
        }, f, ensure_ascii=False, indent=2)
    print(f"Fusion-rule summary CSV: {out_csv}")


if __name__ == "__main__":
    main()
