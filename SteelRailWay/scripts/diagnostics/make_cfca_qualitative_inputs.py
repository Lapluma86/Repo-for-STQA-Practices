#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Prepare a minimal manifest for Cam4 baseline vs CF/CA repair maps."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "thesis_figures" / "cam4_cfca_qualitative"
SCORES = OUT / "scores"


CASES = [
    ("20250417_123456_Cam4_00079", 0, "good", "normal sample"),
    ("20251210_185619_Cam4_00024", 1, "broken", "defect sample"),
    ("20251210_185619_Cam4_00046", 0, "good", "failure sample"),
]


def load_scores(path: Path) -> dict[str, tuple[int, float, int]]:
    rows = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows[row["frame_id"]] = (int(row["rank"]), float(row["score"]), int(row["label"]))
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SCORES.mkdir(parents=True, exist_ok=True)
    (SCORES / "cam4_baseline").mkdir(exist_ok=True)
    (SCORES / "cam4_cfca_repair").mkdir(exist_ok=True)

    shutil.copyfile(
        ROOT
        / "outputs"
        / "rail_peft"
        / "cam4_p1_20260501_225618"
        / "diagnostics"
        / "branch_isolation_server"
        / "cam4_baseline"
        / "result.json",
        SCORES / "cam4_baseline" / "result.json",
    )
    shutil.copyfile(
        ROOT
        / "outputs"
        / "rail_ablation"
        / "cam4_cfca_repair"
        / "cf_ca"
        / "cam4_cf_ca_20260504_170923"
        / "final"
        / "eval"
        / "result.json",
        SCORES / "cam4_cfca_repair" / "result.json",
    )

    baseline_scores = load_scores(
        ROOT
        / "outputs"
        / "rail_peft"
        / "cam4_p1_20260501_225618"
        / "diagnostics"
        / "branch_isolation_server"
        / "cam4_baseline"
        / "scores_fusion.csv"
    )
    repair_scores = load_scores(
        ROOT
        / "outputs"
        / "rail_ablation"
        / "cam4_cfca_repair"
        / "cf_ca"
        / "cam4_cf_ca_20260504_170923"
        / "final"
        / "eval"
        / "scores_fusion.csv"
    )

    rank_summary = OUT / "rank_summary.csv"
    with rank_summary.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "source",
                "baseline_dir",
                "candidate_dir",
                "num_images",
                "num_normal",
                "num_abnormal",
                "total_pairs",
                "baseline_auroc",
                "candidate_auroc",
                "auroc_delta",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "name": "cam4_fusion_cfca",
                "source": "fusion",
                "baseline_dir": "cam4_baseline",
                "candidate_dir": "cam4_cfca_repair",
                "num_images": 18,
                "num_normal": 8,
                "num_abnormal": 10,
                "total_pairs": 80,
                "baseline_auroc": 0.3500,
                "candidate_auroc": 0.7750,
                "auroc_delta": 0.4250,
            }
        )

    manifest = OUT / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "comparison",
            "frame_id",
            "view_id",
            "label",
            "label_name",
            "selection_reasons",
            "baseline_score",
            "candidate_score",
            "delta_score",
            "baseline_rank",
            "candidate_rank",
            "delta_rank",
            "baseline_contribution",
            "candidate_contribution",
            "delta_contribution",
            "positive_flip_count",
            "negative_flip_count",
            "net_flip_count",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for frame_id, label, label_name, reason in CASES:
            b_rank, b_score, _ = baseline_scores[frame_id]
            r_rank, r_score, _ = repair_scores[frame_id]
            writer.writerow(
                {
                    "comparison": "cam4_fusion_cfca",
                    "frame_id": frame_id,
                    "view_id": 4,
                    "label": label,
                    "label_name": label_name,
                    "selection_reasons": reason,
                    "baseline_score": f"{b_score:.8f}",
                    "candidate_score": f"{r_score:.8f}",
                    "delta_score": f"{(r_score - b_score):.8f}",
                    "baseline_rank": b_rank,
                    "candidate_rank": r_rank,
                    "delta_rank": b_rank - r_rank,
                    "baseline_contribution": "",
                    "candidate_contribution": "",
                    "delta_contribution": "",
                    "positive_flip_count": "",
                    "negative_flip_count": "",
                    "net_flip_count": "",
                }
            )

    print(OUT)
    print(manifest)
    print(rank_summary)
    print(SCORES)


if __name__ == "__main__":
    main()
