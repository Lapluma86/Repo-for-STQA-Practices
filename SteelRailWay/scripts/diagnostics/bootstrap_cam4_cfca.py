#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Bootstrap confidence intervals for Cam4 CF/CA ablation results."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

_PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute stratified bootstrap CI for Cam4 CF/CA ablations."
    )
    parser.add_argument("--scores_root", type=str, required=True,
                        help="如 outputs/rail_ablation/cam4_cf_ca_peft")
    parser.add_argument("--summary_csv", type=str, default=None,
                        help="默认读取 <scores_root>/summary.csv")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_csv", type=str, default=None)
    parser.add_argument("--out_json", type=str, default=None)
    return parser


def load_scores_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = list(csv.DictReader(open(path, "r", encoding="utf-8")))
    scores = np.array([float(row["score"]) for row in rows], dtype=np.float64)
    labels = np.array([int(row["label"]) for row in rows], dtype=np.int64)
    return scores, labels


def bootstrap_aurocs(scores: np.ndarray, labels: np.ndarray, iterations: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    normal_idx = np.where(labels == 0)[0]
    abnormal_idx = np.where(labels == 1)[0]
    values = np.zeros(iterations, dtype=np.float64)
    for i in range(iterations):
        sample_normal = rng.choice(normal_idx, size=len(normal_idx), replace=True)
        sample_abnormal = rng.choice(abnormal_idx, size=len(abnormal_idx), replace=True)
        sample_idx = np.concatenate([sample_normal, sample_abnormal])
        sample_scores = scores[sample_idx]
        sample_labels = labels[sample_idx]
        values[i] = roc_auc_score(sample_labels, sample_scores)
    return values


def percentile_ci(values: np.ndarray) -> tuple[float, float]:
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def main() -> None:
    args = build_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    scores_root = Path(args.scores_root)
    summary_csv = Path(args.summary_csv) if args.summary_csv else scores_root / "summary.csv"
    rows = list(csv.DictReader(open(summary_csv, "r", encoding="utf-8")))
    rows_by_mode = {row["module_ablation"]: row for row in rows}
    if "full" not in rows_by_mode:
        raise RuntimeError("summary.csv 中必须包含 full 行")

    full_scores, full_labels = load_scores_csv(scores_root / "full" / "scores_fusion.csv")
    full_boot = bootstrap_aurocs(full_scores, full_labels, args.iterations, args.seed)
    full_ci = percentile_ci(full_boot)
    full_mean = float(roc_auc_score(full_labels, full_scores))

    out_rows = []
    for idx, mode in enumerate(["full", "no_cf", "no_ca", "no_cf_ca"]):
        mode_dir = scores_root / mode
        scores, labels = load_scores_csv(mode_dir / "scores_fusion.csv")
        point = float(roc_auc_score(labels, scores))
        boot = bootstrap_aurocs(scores, labels, args.iterations, args.seed + idx + 1)
        low, high = percentile_ci(boot)
        delta_boot = boot - full_boot if mode != "full" else np.zeros_like(boot)
        delta_low, delta_high = percentile_ci(delta_boot)
        out_rows.append({
            "module_ablation": mode,
            "fusion_auroc": f"{point:.8f}",
            "fusion_ci_low": f"{low:.8f}",
            "fusion_ci_high": f"{high:.8f}",
            "delta_vs_full": f"{(point - full_mean):.8f}",
            "delta_ci_low": f"{delta_low:.8f}",
            "delta_ci_high": f"{delta_high:.8f}",
            "iterations": str(args.iterations),
        })

    out_csv = Path(args.out_csv) if args.out_csv else scores_root / "bootstrap_summary.csv"
    out_json = Path(args.out_json) if args.out_json else scores_root / "bootstrap_summary.json"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "scores_root": str(scores_root),
            "iterations": args.iterations,
            "rows": out_rows,
        }, f, ensure_ascii=False, indent=2)
    print(f"Bootstrap summary CSV: {out_csv}")
    print(f"Bootstrap summary JSON: {out_json}")


if __name__ == "__main__":
    main()
