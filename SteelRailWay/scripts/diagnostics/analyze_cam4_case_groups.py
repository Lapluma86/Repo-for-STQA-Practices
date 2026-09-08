#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Group-level case analysis for Cam4 PEFT CF/CA ablations."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

_PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))


ALLOWED_GROUPS = {"rgb_only", "depth_only", "cross_modal", "uncertain"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze Cam4 case-group score deltas for full vs no_cf/no_ca."
    )
    parser.add_argument("--scores_root", type=str, default="outputs/rail_ablation/cam4_cf_ca_peft")
    parser.add_argument("--frame_groups", type=str,
                        default="outputs/rail_ablation/cam4_case_groups/frame_groups.csv")
    parser.add_argument("--out_dir", type=str, default="outputs/rail_ablation/cam4_case_groups/analysis")
    return parser


def load_scores(path: Path) -> dict[str, dict[str, str]]:
    return {row["frame_id"]: row for row in csv.DictReader(open(path, "r", encoding="utf-8"))}


def median(values: list[float]) -> float:
    return float(np.median(np.array(values, dtype=np.float64))) if values else 0.0


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    group_rows = list(csv.DictReader(open(args.frame_groups, "r", encoding="utf-8")))
    for row in group_rows:
        if row["anomaly_group"] not in ALLOWED_GROUPS:
            raise ValueError(f"invalid anomaly_group: {row['anomaly_group']}")

    full_scores = load_scores(Path(args.scores_root) / "full" / "scores_fusion.csv")
    no_cf_scores = load_scores(Path(args.scores_root) / "no_cf" / "scores_fusion.csv")
    no_ca_scores = load_scores(Path(args.scores_root) / "no_ca" / "scores_fusion.csv")

    detailed_rows = []
    grouped = {
        "full_vs_no_cf": defaultdict(list),
        "full_vs_no_ca": defaultdict(list),
    }

    for row in group_rows:
        frame_id = row["frame_id"]
        if int(row["label"]) != 1:
            continue
        delta_no_cf = float(no_cf_scores[frame_id]["score"]) - float(full_scores[frame_id]["score"])
        delta_no_ca = float(no_ca_scores[frame_id]["score"]) - float(full_scores[frame_id]["score"])
        detailed_rows.append({
            "frame_id": frame_id,
            "label": row["label"],
            "anomaly_group": row["anomaly_group"],
            "notes": row.get("notes", ""),
            "full_score": full_scores[frame_id]["score"],
            "no_cf_score": no_cf_scores[frame_id]["score"],
            "no_ca_score": no_ca_scores[frame_id]["score"],
            "delta_full_vs_no_cf": f"{delta_no_cf:.8f}",
            "delta_full_vs_no_ca": f"{delta_no_ca:.8f}",
        })
        grouped["full_vs_no_cf"][row["anomaly_group"]].append(delta_no_cf)
        grouped["full_vs_no_ca"][row["anomaly_group"]].append(delta_no_ca)

    summary_rows = []
    for comparison, by_group in grouped.items():
        for group_name in ["rgb_only", "depth_only", "cross_modal", "uncertain"]:
            values = by_group.get(group_name, [])
            summary_rows.append({
                "comparison": comparison,
                "anomaly_group": group_name,
                "count": len(values),
                "median_score_delta": f"{median(values):.8f}",
                "mean_score_delta": f"{(float(np.mean(values)) if values else 0.0):.8f}",
            })

    with open(out_dir / "frame_group_deltas.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(detailed_rows[0].keys()) if detailed_rows else [
            "frame_id", "label", "anomaly_group", "notes", "full_score", "no_cf_score", "no_ca_score",
            "delta_full_vs_no_cf", "delta_full_vs_no_ca",
        ])
        writer.writeheader()
        writer.writerows(detailed_rows)

    with open(out_dir / "group_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()) if summary_rows else [
            "comparison", "anomaly_group", "count", "median_score_delta", "mean_score_delta",
        ])
        writer.writeheader()
        writer.writerows(summary_rows)

    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("Cam4 case-group analysis\n")
        f.write(f"created_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"scores_root: {args.scores_root}\n")
        f.write(f"frame_groups: {args.frame_groups}\n\n")
        for row in summary_rows:
            f.write(
                f"{row['comparison']} | {row['anomaly_group']}: "
                f"count={row['count']} median={row['median_score_delta']} mean={row['mean_score_delta']}\n"
            )

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "detailed_rows": detailed_rows,
            "summary_rows": summary_rows,
        }, f, ensure_ascii=False, indent=2)

    print(f"Case-group analysis written to: {out_dir}")


if __name__ == "__main__":
    main()
