#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Analyze per-image and pairwise ranking changes between baseline and PEFT runs."""

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
import json
from dataclasses import dataclass
from datetime import datetime


DEFAULT_COMPARISONS = [
    {
        "name": "cam4_depth",
        "baseline_dir": "cam4_baseline",
        "candidate_dir": "cam4_with_cam4peft",
        "source": "depth",
    },
    {
        "name": "cam4_depth_isolated",
        "baseline_dir": "cam4_baseline",
        "candidate_dir": "cam4_with_cam4peft",
        "source": "depth_isolated",
    },
    {
        "name": "cam5_fusion",
        "baseline_dir": "cam5_baseline",
        "candidate_dir": "cam5_with_cam4peft",
        "source": "fusion",
    },
]


@dataclass
class ScoreRow:
    frame_id: str
    rank: int
    score: float
    label: int


def build_parser():
    parser = argparse.ArgumentParser(
        description="Analyze baseline vs PEFT rank changes from saved scores CSV files."
    )
    parser.add_argument(
        "--scores_root",
        type=str,
        default="outputs/rail_peft/cam4_p1_20260501_225618/diagnostics/branch_isolation_server",
        help="包含 cam*_baseline / cam*_with_cam4peft 分数目录的根目录",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="输出目录，默认写到 <scores_root>_rank_analysis",
    )
    parser.add_argument(
        "--comparisons",
        type=str,
        default=None,
        help="可选：JSON 文件或 JSON 字符串，元素包含 name/baseline_dir/candidate_dir/source",
    )
    return parser


def default_out_dir(scores_root: Path) -> Path:
    if scores_root.name.endswith("_server"):
        return scores_root.parent / f"{scores_root.name}_rank_analysis"
    return scores_root.parent / f"{scores_root.name}_rank_analysis"


def load_comparisons(value: str | None):
    if not value:
        return DEFAULT_COMPARISONS
    path = Path(value)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    else:
        raw = json.loads(value)
    if not isinstance(raw, list):
        raise TypeError("comparisons must be a JSON list.")
    return raw


def load_score_rows(path: Path) -> dict[str, ScoreRow]:
    rows = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item = ScoreRow(
                frame_id=row["frame_id"],
                rank=int(row["rank"]),
                score=float(row["score"]),
                label=int(row["label"]),
            )
            rows[item.frame_id] = item
    return rows


def compute_auc_contribution(rows: dict[str, ScoreRow]) -> dict[str, float]:
    normals = [r for r in rows.values() if r.label == 0]
    abnormals = [r for r in rows.values() if r.label == 1]
    contrib = {}
    for frame_id, row in rows.items():
        if row.label == 1:
            wins = sum(1 for n in normals if row.score > n.score)
            ties = sum(1 for n in normals if row.score == n.score)
            contrib[frame_id] = (wins + 0.5 * ties) / len(normals) if normals else 0.0
        else:
            wins = sum(1 for a in abnormals if a.score > row.score)
            ties = sum(1 for a in abnormals if a.score == row.score)
            contrib[frame_id] = (wins + 0.5 * ties) / len(abnormals) if abnormals else 0.0
    return contrib


def compute_auroc_from_contrib(rows: dict[str, ScoreRow], contrib: dict[str, float]) -> float:
    abnormals = [frame_id for frame_id, row in rows.items() if row.label == 1]
    if not abnormals:
        return 0.0
    return sum(contrib[frame_id] for frame_id in abnormals) / len(abnormals)


def median(values: list[float]) -> float:
    values = sorted(values)
    n = len(values)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 1:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0


def compute_class_stats(rows: dict[str, ScoreRow]) -> dict[str, float]:
    normals = [r.score for r in rows.values() if r.label == 0]
    abnormals = [r.score for r in rows.values() if r.label == 1]
    return {
        "normal_mean": sum(normals) / len(normals) if normals else 0.0,
        "normal_median": median(normals),
        "abnormal_mean": sum(abnormals) / len(abnormals) if abnormals else 0.0,
        "abnormal_median": median(abnormals),
    }


def analyze_comparison(scores_root: Path, out_dir: Path, spec: dict) -> dict:
    name = spec["name"]
    source = spec["source"]
    baseline_dir = scores_root / spec["baseline_dir"]
    candidate_dir = scores_root / spec["candidate_dir"]
    baseline_path = baseline_dir / f"scores_{source}.csv"
    candidate_path = candidate_dir / f"scores_{source}.csv"
    if not baseline_path.exists():
        raise FileNotFoundError(f"Missing baseline scores: {baseline_path}")
    if not candidate_path.exists():
        raise FileNotFoundError(f"Missing candidate scores: {candidate_path}")

    baseline_rows = load_score_rows(baseline_path)
    candidate_rows = load_score_rows(candidate_path)
    if set(baseline_rows.keys()) != set(candidate_rows.keys()):
        raise RuntimeError(f"Mismatched frame ids for comparison: {name}")

    baseline_contrib = compute_auc_contribution(baseline_rows)
    candidate_contrib = compute_auc_contribution(candidate_rows)
    baseline_stats = compute_class_stats(baseline_rows)
    candidate_stats = compute_class_stats(candidate_rows)

    normals = [frame_id for frame_id, row in baseline_rows.items() if row.label == 0]
    abnormals = [frame_id for frame_id, row in baseline_rows.items() if row.label == 1]

    frame_rows = []
    frame_flip_counts = {frame_id: {"positive": 0, "negative": 0} for frame_id in baseline_rows}
    positive_flips = []
    negative_flips = []

    for abnormal_id in abnormals:
        for normal_id in normals:
            base_ok = baseline_rows[abnormal_id].score > baseline_rows[normal_id].score
            cand_ok = candidate_rows[abnormal_id].score > candidate_rows[normal_id].score
            flip_row = {
                "abnormal_frame_id": abnormal_id,
                "normal_frame_id": normal_id,
                "baseline_abnormal_score": f"{baseline_rows[abnormal_id].score:.8f}",
                "baseline_normal_score": f"{baseline_rows[normal_id].score:.8f}",
                "candidate_abnormal_score": f"{candidate_rows[abnormal_id].score:.8f}",
                "candidate_normal_score": f"{candidate_rows[normal_id].score:.8f}",
            }
            if (not base_ok) and cand_ok:
                positive_flips.append(flip_row)
                frame_flip_counts[abnormal_id]["positive"] += 1
                frame_flip_counts[normal_id]["positive"] += 1
            elif base_ok and (not cand_ok):
                negative_flips.append(flip_row)
                frame_flip_counts[abnormal_id]["negative"] += 1
                frame_flip_counts[normal_id]["negative"] += 1

    for frame_id in sorted(baseline_rows.keys()):
        base_row = baseline_rows[frame_id]
        cand_row = candidate_rows[frame_id]
        pos_count = frame_flip_counts[frame_id]["positive"]
        neg_count = frame_flip_counts[frame_id]["negative"]
        frame_rows.append({
            "frame_id": frame_id,
            "label": base_row.label,
            "baseline_score": f"{base_row.score:.8f}",
            "candidate_score": f"{cand_row.score:.8f}",
            "delta_score": f"{(cand_row.score - base_row.score):.8f}",
            "baseline_rank": base_row.rank,
            "candidate_rank": cand_row.rank,
            "delta_rank": base_row.rank - cand_row.rank,
            "baseline_contribution": f"{baseline_contrib[frame_id]:.8f}",
            "candidate_contribution": f"{candidate_contrib[frame_id]:.8f}",
            "delta_contribution": f"{(candidate_contrib[frame_id] - baseline_contrib[frame_id]):.8f}",
            "positive_flip_count": pos_count,
            "negative_flip_count": neg_count,
            "net_flip_count": pos_count - neg_count,
        })

    comparison_dir = out_dir / name
    comparison_dir.mkdir(parents=True, exist_ok=True)

    def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    write_csv(
        comparison_dir / "frame_deltas.csv",
        frame_rows,
        [
            "frame_id",
            "label",
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
        ],
    )
    write_csv(
        comparison_dir / "pair_flips_positive.csv",
        positive_flips,
        [
            "abnormal_frame_id",
            "normal_frame_id",
            "baseline_abnormal_score",
            "baseline_normal_score",
            "candidate_abnormal_score",
            "candidate_normal_score",
        ],
    )
    write_csv(
        comparison_dir / "pair_flips_negative.csv",
        negative_flips,
        [
            "abnormal_frame_id",
            "normal_frame_id",
            "baseline_abnormal_score",
            "baseline_normal_score",
            "candidate_abnormal_score",
            "candidate_normal_score",
        ],
    )

    frame_rows_by_delta = sorted(frame_rows, key=lambda x: float(x["delta_contribution"]), reverse=True)
    positive_candidates = [item for item in frame_rows_by_delta if float(item["delta_contribution"]) > 0]
    negative_candidates = [item for item in frame_rows_by_delta if float(item["delta_contribution"]) < 0]
    top_positive = positive_candidates[:5]
    top_negative = sorted(negative_candidates, key=lambda x: float(x["delta_contribution"]))[:5]

    with open(comparison_dir / "summary.txt", "w", encoding="utf-8") as f:
        base_auc = compute_auroc_from_contrib(baseline_rows, baseline_contrib)
        cand_auc = compute_auroc_from_contrib(candidate_rows, candidate_contrib)
        f.write(f"Comparison: {name}\n")
        f.write(f"Source: {source}\n")
        f.write(f"Baseline dir: {spec['baseline_dir']}\n")
        f.write(f"Candidate dir: {spec['candidate_dir']}\n")
        f.write(f"Baseline AUROC: {base_auc:.8f}\n")
        f.write(f"Candidate AUROC: {cand_auc:.8f}\n")
        f.write(f"AUROC delta: {cand_auc - base_auc:.8f}\n")
        f.write(f"Positive pair flips: {len(positive_flips)}\n")
        f.write(f"Negative pair flips: {len(negative_flips)}\n")
        f.write(f"Net pair flips: {len(positive_flips) - len(negative_flips)}\n\n")
        f.write(
            "Baseline class stats: "
            f"normal_mean={baseline_stats['normal_mean']:.8f}, "
            f"normal_median={baseline_stats['normal_median']:.8f}, "
            f"abnormal_mean={baseline_stats['abnormal_mean']:.8f}, "
            f"abnormal_median={baseline_stats['abnormal_median']:.8f}\n"
        )
        f.write(
            "Candidate class stats: "
            f"normal_mean={candidate_stats['normal_mean']:.8f}, "
            f"normal_median={candidate_stats['normal_median']:.8f}, "
            f"abnormal_mean={candidate_stats['abnormal_mean']:.8f}, "
            f"abnormal_median={candidate_stats['abnormal_median']:.8f}\n\n"
        )
        f.write("Top positive frames by delta_contribution:\n")
        if top_positive:
            for row in top_positive:
                f.write(
                    f"  {row['frame_id']} label={row['label']} "
                    f"delta_contribution={row['delta_contribution']} "
                    f"net_flip_count={row['net_flip_count']} "
                    f"baseline_rank={row['baseline_rank']} candidate_rank={row['candidate_rank']}\n"
                )
        else:
            f.write("  N/A\n")
        f.write("\nTop negative frames by delta_contribution:\n")
        if top_negative:
            for row in top_negative:
                f.write(
                    f"  {row['frame_id']} label={row['label']} "
                    f"delta_contribution={row['delta_contribution']} "
                    f"net_flip_count={row['net_flip_count']} "
                    f"baseline_rank={row['baseline_rank']} candidate_rank={row['candidate_rank']}\n"
                )
        else:
            f.write("  N/A\n")

    summary = {
        "name": name,
        "source": source,
        "baseline_dir": spec["baseline_dir"],
        "candidate_dir": spec["candidate_dir"],
        "num_images": len(frame_rows),
        "num_normal": len(normals),
        "num_abnormal": len(abnormals),
        "total_pairs": len(normals) * len(abnormals),
        "baseline_auroc": compute_auroc_from_contrib(baseline_rows, baseline_contrib),
        "candidate_auroc": compute_auroc_from_contrib(candidate_rows, candidate_contrib),
        "auroc_delta": compute_auroc_from_contrib(candidate_rows, candidate_contrib)
        - compute_auroc_from_contrib(baseline_rows, baseline_contrib),
        "positive_pair_flips": len(positive_flips),
        "negative_pair_flips": len(negative_flips),
        "net_pair_flips": len(positive_flips) - len(negative_flips),
        "top_positive_frame": top_positive[0]["frame_id"] if top_positive else "",
        "top_positive_label": top_positive[0]["label"] if top_positive else "",
        "top_positive_delta_contribution": float(top_positive[0]["delta_contribution"]) if top_positive else 0.0,
        "top_positive_net_flip_count": top_positive[0]["net_flip_count"] if top_positive else 0,
        "top_negative_frame": top_negative[0]["frame_id"] if top_negative else "",
        "top_negative_label": top_negative[0]["label"] if top_negative else "",
        "top_negative_delta_contribution": float(top_negative[0]["delta_contribution"]) if top_negative else 0.0,
        "top_negative_net_flip_count": top_negative[0]["net_flip_count"] if top_negative else 0,
        "baseline_normal_mean": baseline_stats["normal_mean"],
        "baseline_abnormal_mean": baseline_stats["abnormal_mean"],
        "candidate_normal_mean": candidate_stats["normal_mean"],
        "candidate_abnormal_mean": candidate_stats["abnormal_mean"],
    }
    return summary


def main():
    args = build_parser().parse_args()
    scores_root = Path(args.scores_root)
    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir(scores_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    comparisons = load_comparisons(args.comparisons)
    rows = [analyze_comparison(scores_root, out_dir, spec) for spec in comparisons]

    summary_csv = out_dir / "summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
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
                "positive_pair_flips",
                "negative_pair_flips",
                "net_pair_flips",
                "top_positive_frame",
                "top_positive_label",
                "top_positive_delta_contribution",
                "top_positive_net_flip_count",
                "top_negative_frame",
                "top_negative_label",
                "top_negative_delta_contribution",
                "top_negative_net_flip_count",
                "baseline_normal_mean",
                "baseline_abnormal_mean",
                "candidate_normal_mean",
                "candidate_abnormal_mean",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary_txt = out_dir / "summary.txt"
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("Branch rank analysis\n")
        f.write(f"created_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"scores_root: {scores_root}\n")
        f.write(f"comparisons: {len(comparisons)}\n\n")
        for row in rows:
            f.write(
                f"{row['name']} ({row['source']}): "
                f"baseline={row['baseline_auroc']:.8f}, "
                f"candidate={row['candidate_auroc']:.8f}, "
                f"delta={row['auroc_delta']:.8f}, "
                f"pair_flips=+{row['positive_pair_flips']}/-{row['negative_pair_flips']} "
                f"(net {row['net_pair_flips']})\n"
            )
            f.write(
                f"  top_positive_frame={row['top_positive_frame']} "
                f"(label={row['top_positive_label']}, delta_contribution={row['top_positive_delta_contribution']:.8f}, "
                f"net_flip_count={row['top_positive_net_flip_count']})\n"
            )
            f.write(
                f"  top_negative_frame={row['top_negative_frame']} "
                f"(label={row['top_negative_label']}, delta_contribution={row['top_negative_delta_contribution']:.8f}, "
                f"net_flip_count={row['top_negative_net_flip_count']})\n"
            )
            f.write("\n")

    print(f"Rank-analysis summary CSV: {summary_csv}")
    print(f"Rank-analysis summary TXT: {summary_txt}")


if __name__ == "__main__":
    main()
