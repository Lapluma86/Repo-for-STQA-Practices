#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Summarize image-level false positives from saved score CSV files.

The thesis uses a safety-first operating point: FP@R100.  The threshold is set
to the minimum anomaly score among positive samples, and samples with score
greater than or equal to that threshold are predicted as abnormal.  This gives
100% image-level recall whenever both classes are present, then counts how many
normal images would be sent to manual review.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def rel(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def read_scores(path: str | Path) -> list[dict[str, str]]:
    score_path = rel(path)
    if not score_path.exists():
        raise FileNotFoundError(score_path)
    with score_path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def fp_at_recall100(path: str | Path) -> dict[str, str]:
    rows = read_scores(path)
    parsed = []
    for row in rows:
        parsed.append({
            "frame_id": row.get("frame_id", ""),
            "score": float(row["score"]),
            "label": int(float(row["label"])),
        })

    positives = [row for row in parsed if row["label"] == 1]
    negatives = [row for row in parsed if row["label"] == 0]
    if not positives or not negatives:
        return {
            "n_images": str(len(parsed)),
            "n_normal": str(len(negatives)),
            "n_abnormal": str(len(positives)),
            "threshold_r100": "",
            "fp_r100": "",
            "fpr_r100": "",
            "precision_r100": "",
            "fp_frames": "",
        }

    threshold = min(row["score"] for row in positives)
    fp_rows = [row for row in negatives if row["score"] >= threshold]
    fp_count = len(fp_rows)
    fpr = fp_count / len(negatives)
    precision = len(positives) / (len(positives) + fp_count)
    fp_frames = " ".join(row["frame_id"] for row in sorted(fp_rows, key=lambda r: -r["score"]))

    return {
        "n_images": str(len(parsed)),
        "n_normal": str(len(negatives)),
        "n_abnormal": str(len(positives)),
        "threshold_r100": f"{threshold:.8f}",
        "fp_r100": str(fp_count),
        "fpr_r100": f"{fpr:.4f}",
        "precision_r100": f"{precision:.4f}",
        "fp_frames": fp_frames,
    }


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "group",
        "scheme",
        "score_source",
        "scores_csv",
        "n_images",
        "n_normal",
        "n_abnormal",
        "threshold_r100",
        "fp_r100",
        "fpr_r100",
        "precision_r100",
        "fp_frames",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def collect_rows() -> list[dict[str, str]]:
    specs: list[tuple[str, str, str, str]] = []

    # Static baselines in Table 4-1.
    for method in ("padim", "patchcore", "fastflow", "draem", "stfpm", "rd4ad_i"):
        specs.append((
            "static_rgb_cam4",
            method,
            "rgb",
            f"results/baselines_rgb/cam4/{method}/scores.csv",
        ))
    for method in ("padim", "patchcore", "fastflow", "draem", "stfpm", "rd4ad_i"):
        specs.append((
            "static_rgbd_cam4",
            method,
            "fusion",
            f"results/baselines_rgbd/cam4/{method}/scores_fusion.csv",
        ))

    # Main deployed TRD adaptation in Table 4-1A.
    specs.extend([
        (
            "trd_cam4_adaptation",
            "baseline_trd",
            "fusion",
            "outputs/rail_peft/cam4_p1_20260501_225618/diagnostics/branch_auc/cam4_baseline/scores_fusion.csv",
        ),
        (
            "trd_cam4_adaptation",
            "depth_affine_peft",
            "fusion",
            "outputs/rail_peft/cam4_p1_20260501_225618/diagnostics/branch_auc/cam4_with_cam4peft/scores_fusion.csv",
        ),
        (
            "trd_cam4_adaptation",
            "depth_affine_peft_cfca_repair",
            "fusion",
            "outputs/rail_ablation/cam4_cfca_repair/cf_ca/cam4_cf_ca_20260504_170923/final/eval/scores_fusion.csv",
        ),
    ])

    # CF/CA repair scopes in Table 4-2.
    specs.extend([
        (
            "cfca_repair_scope",
            "cf_only",
            "fusion",
            "outputs/rail_ablation/cam4_cfca_repair/cf_only/cam4_cf_only_20260504_154942/final/eval/scores_fusion.csv",
        ),
        (
            "cfca_repair_scope",
            "ca_only",
            "fusion",
            "outputs/rail_ablation/cam4_cfca_repair/ca_only/cam4_ca_only_20260504_163147/final/eval/scores_fusion.csv",
        ),
        (
            "cfca_repair_scope",
            "cf_ca",
            "fusion",
            "outputs/rail_ablation/cam4_cfca_repair/cf_ca/cam4_cf_ca_20260504_170923/final/eval/scores_fusion.csv",
        ),
    ])

    # Continuation comparison in Table 4-2A.
    specs.extend([
        (
            "continuation_comparison",
            "cf_ca_repair",
            "fusion",
            "outputs/rail_ablation/cam4_cfca_repair/cf_ca/cam4_cf_ca_20260504_170923/final/eval/scores_fusion.csv",
        ),
        (
            "continuation_comparison",
            "peft_full_then_cf_ca",
            "fusion",
            "outputs/rail_ablation/cam4_peft_full_then_cf_ca/cam4_peft_full_then_cf_ca_20260504_200444/peft_full_then_cf_ca/final/eval/scores_fusion.csv",
        ),
    ])

    # CAD / cross-task PEFT mounting in Tables 4-4 and 4-5.
    for cam in (1, 5):
        specs.extend([
            (
                "cad_cross_task",
                f"cam{cam}_baseline",
                "fusion",
                f"results/cam4_cfca_repair_cross_camera_20260504_225252/cam{cam}_baseline/scores_fusion.csv",
            ),
            (
                "cad_cross_task",
                f"cam{cam}_with_cam4peft",
                "fusion",
                f"results/cam4_cfca_repair_cross_camera_20260504_225252/cam{cam}_with_cam4peft/scores_fusion.csv",
            ),
        ])

    rows = []
    for group, scheme, source, path in specs:
        metrics = fp_at_recall100(path)
        rows.append({
            "group": group,
            "scheme": scheme,
            "score_source": source,
            "scores_csv": str(path),
            **metrics,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize FP@R100 from saved image score CSV files.")
    parser.add_argument("--out_dir", default="outputs/fp_analysis")
    args = parser.parse_args()

    rows = collect_rows()
    out_dir = rel(args.out_dir)
    write_csv(out_dir / "all_fp_summary.csv", rows)
    for group in sorted({row["group"] for row in rows}):
        write_csv(out_dir / f"{group}_fp.csv", [row for row in rows if row["group"] == group])
    print(f"Wrote FP summaries to: {out_dir}")


if __name__ == "__main__":
    main()
