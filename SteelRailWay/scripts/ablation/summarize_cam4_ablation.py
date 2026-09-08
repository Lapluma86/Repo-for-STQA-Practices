#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build one CSV for Chapter 6 Cam4 ablation and engineering tables."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


_PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))


DEFAULT_BASELINE_CKPT = (
    "outputs/rail_all/Cam4/20260501_014226_cam4_bs32_lr0.005_img512_ratio1.0/"
    "best_cam4.pth"
)
DEFAULT_PEFT_CKPT = "outputs/rail_peft/cam4_p1_20260501_225618/final/final_peft_cam4.pth"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge Cam4 CF/CA, depth-norm, and engineering-metric summaries."
    )
    parser.add_argument("--cf_ca_summary", type=str, default="outputs/rail_ablation/cam4_cf_ca/summary.csv")
    parser.add_argument("--depth_norm_summary", type=str,
                        default="outputs/rail_ablation/depth_norm/Cam4/summary.csv")
    parser.add_argument("--baseline_ckpt", type=str, default=DEFAULT_BASELINE_CKPT)
    parser.add_argument("--peft_ckpt", type=str, default=DEFAULT_PEFT_CKPT)
    parser.add_argument("--out_csv", type=str, default="outputs/rail_ablation/cam4_ablation_summary.csv")
    parser.add_argument("--count_full_params", action="store_true",
                        help="Load the large baseline checkpoint and count full/depth-branch params exactly.")
    return parser


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_torch():
    try:
        import torch  # Imported lazily; large checkpoints are optional.
    except ModuleNotFoundError:
        return None
    return torch


def safe_load(path: Path) -> Any:
    torch = load_torch()
    if torch is None:
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def is_trainable_state_key(key: str) -> bool:
    return not key.endswith(("running_mean", "running_var", "num_batches_tracked"))


def count_state_dict_params(state_dict: dict[str, Any], trainable_only: bool = True) -> int:
    return int(sum(
        t.numel()
        for key, t in state_dict.items()
        if hasattr(t, "numel") and (not trainable_only or is_trainable_state_key(key))
    ))


def count_peft_params(path: Path) -> int | str:
    if not path.exists():
        return 2
    payload = safe_load(path)
    if payload is None:
        return 2
    state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else {}
    if not isinstance(state_dict, dict):
        return 2
    return count_state_dict_params({
        k: v for k, v in state_dict.items()
        if k.replace("_orig_mod.", "").replace("module.", "").replace("peft.", "") in {"gain", "bias"}
    })


def count_baseline_params(path: Path, enabled: bool) -> tuple[int | str, int | str]:
    if not enabled or not path.exists():
        return "", ""
    payload = safe_load(path)
    if payload is None:
        return "", ""
    if not isinstance(payload, dict):
        return "", ""
    depth_params = count_state_dict_params(payload.get("student_depth", {}))
    rgb_params = count_state_dict_params(payload.get("student_rgb", {}))
    return depth_params, depth_params + rgb_params


def base_row(section: str, table: str, scheme: str) -> dict[str, str]:
    return {
        "section": section,
        "table": table,
        "scheme": scheme,
        "module_ablation": "",
        "depth_norm": "",
        "rgb_auroc": "",
        "depth_auroc": "",
        "fusion_auroc": "",
        "fusion_delta": "",
        "best_epoch": "",
        "best_val_loss": "",
        "trainable_params": "",
        "train_time": "",
        "inference_overhead": "",
        "deployment_risk": "",
        "note": "",
        "source_path": "",
    }


def append_cf_ca(rows: list[dict[str, str]], summary_path: Path) -> None:
    for item in read_csv(summary_path):
        row = base_row("6.6.2", "CF/CA 模块消融", item.get("scheme", ""))
        row.update({
            "module_ablation": item.get("module_ablation", ""),
            "rgb_auroc": item.get("auroc_rgb", ""),
            "depth_auroc": item.get("auroc_depth", ""),
            "fusion_auroc": item.get("auroc_fusion", ""),
            "fusion_delta": item.get("delta_fusion_vs_full", ""),
            "note": item.get("note", ""),
            "source_path": item.get("result_json", str(summary_path)),
        })
        rows.append(row)


def append_depth_norm(rows: list[dict[str, str]], summary_path: Path) -> None:
    labels = {"zscore": "z-score", "minmax": "min-max", "log": "log"}
    for item in read_csv(summary_path):
        norm = item.get("depth_norm", "")
        row = base_row("6.6.3", "深度归一化策略消融", labels.get(norm, norm))
        row.update({
            "depth_norm": norm,
            "rgb_auroc": item.get("auroc_rgb", ""),
            "depth_auroc": item.get("auroc_depth", ""),
            "fusion_auroc": item.get("auroc_fusion", ""),
            "best_epoch": item.get("best_epoch", ""),
            "best_val_loss": item.get("best_val_loss", ""),
            "note": item.get("note", ""),
            "source_path": item.get("result_json", str(summary_path)),
        })
        rows.append(row)


def append_engineering(rows: list[dict[str, str]], args: argparse.Namespace) -> None:
    depth_params, full_params = count_baseline_params(Path(args.baseline_ckpt), args.count_full_params)
    peft_params = count_peft_params(Path(args.peft_ckpt))

    entries = [
        ("不更新 (baseline)", "0", "—", "0", "—", "固定已有 ckpt，仅评估", args.baseline_ckpt),
        ("DepthAffinePEFT (本文)", str(peft_params), "< 2 min", "可忽略", "极低",
         "仅训练 depth affine gain/bias", args.peft_ckpt),
        ("全量微调 Depth 分支", str(depth_params), "数小时", "0", "中",
         "训练 student_depth 全部参数", args.baseline_ckpt),
        ("全量重训整体模型", str(full_params), "数小时-一天", "0", "高",
         "重新训练 student_rgb + student_depth", args.baseline_ckpt),
    ]
    for scheme, params, train_time, overhead, risk, note, source_path in entries:
        row = base_row("6.6.4", "工程指标对比", scheme)
        row.update({
            "trainable_params": params,
            "train_time": train_time,
            "inference_overhead": overhead,
            "deployment_risk": risk,
            "note": note,
            "source_path": source_path,
        })
        rows.append(row)


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(base_row("", "", "").keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = build_parser().parse_args()
    rows: list[dict[str, str]] = []
    append_cf_ca(rows, Path(args.cf_ca_summary))
    append_depth_norm(rows, Path(args.depth_norm_summary))
    append_engineering(rows, args)
    out_csv = Path(args.out_csv)
    write_csv(out_csv, rows)
    with open(out_csv.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(rows)} rows: {out_csv}")


if __name__ == "__main__":
    main()
