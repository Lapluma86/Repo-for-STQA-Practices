#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Export key RGB/depth frames referenced by rank-analysis outputs."""

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
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from PIL import Image

try:
    import tifffile
except ImportError:  # pragma: no cover
    tifffile = None


DEFAULT_MANUAL_FRAMES = {
    "cam4_depth": [
        "20250417_123456_Cam4_00079",
        "20251210_185619_Cam4_00024",
        "20251210_185619_Cam4_00046",
    ],
    "cam4_depth_isolated": [
        "20250417_123456_Cam4_00079",
        "20251210_185619_Cam4_00024",
        "20251210_185619_Cam4_00046",
    ],
    "cam5_fusion": [
        "20251112_191827_Cam5_00067",
        "20251112_191827_Cam5_00123",
        "20251112_191827_Cam5_00022",
        "20251112_191827_Cam5_00040",
    ],
}


@dataclass
class FrameRecord:
    frame_id: str
    label: int
    baseline_score: float
    candidate_score: float
    delta_score: float
    baseline_rank: int
    candidate_rank: int
    delta_rank: int
    baseline_contribution: float
    candidate_contribution: float
    delta_contribution: float
    positive_flip_count: int
    negative_flip_count: int
    net_flip_count: int


def build_parser():
    parser = argparse.ArgumentParser(
        description="Export key RGB/depth frames from branch rank-analysis outputs."
    )
    parser.add_argument(
        "--rank_root",
        type=str,
        default="outputs/rail_peft/cam4_p1_20260501_225618/diagnostics/branch_rank_analysis_server",
        help="包含 cam*_*/frame_deltas.csv 的 rank-analysis 目录",
    )
    parser.add_argument(
        "--test_root",
        type=str,
        default="rail_mvtec_gt_test",
        help="测试集根目录，内部应包含 rail_mvtec/ 与 rail_mvtec_depth/",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="输出目录，默认写到 <rank_root> 同级的 key_frames[_server]",
    )
    parser.add_argument(
        "--comparisons",
        type=str,
        nargs="*",
        default=None,
        help="仅导出指定 comparison 名称；默认扫描 rank_root 下全部 comparison 子目录",
    )
    parser.add_argument(
        "--top_positive",
        type=int,
        default=3,
        help="每个 comparison 取 delta_contribution 最高的前 N 帧",
    )
    parser.add_argument(
        "--top_negative",
        type=int,
        default=2,
        help="每个 comparison 取 delta_contribution 最低的前 N 帧",
    )
    parser.add_argument(
        "--top_positive_pairs",
        type=int,
        default=5,
        help="每个 comparison 从正向 pair flip 里额外纳入前 N 行的 abnormal/normal 端点",
    )
    parser.add_argument(
        "--top_negative_pairs",
        type=int,
        default=3,
        help="每个 comparison 从负向 pair flip 里额外纳入前 N 行的 abnormal/normal 端点",
    )
    parser.add_argument(
        "--manual_frames",
        type=str,
        default=None,
        help="可选：JSON 文件或 JSON 字符串，comparison -> [frame_id,...]；默认使用内置关键帧",
    )
    parser.add_argument(
        "--copy_mode",
        type=str,
        choices=["copy", "hardlink"],
        default="copy",
        help="原图导出方式，默认 copy；同盘场景可选 hardlink",
    )
    parser.add_argument(
        "--depth_pct_low",
        type=float,
        default=1.0,
        help="depth 预览下百分位",
    )
    parser.add_argument(
        "--depth_pct_high",
        type=float,
        default=99.0,
        help="depth 预览上百分位",
    )
    return parser


def default_out_dir(rank_root: Path) -> Path:
    name = "key_frames_server" if rank_root.name.endswith("_server") else "key_frames"
    return rank_root.parent / name


def load_manual_frames(value: str | None) -> dict[str, list[str]]:
    if not value:
        return DEFAULT_MANUAL_FRAMES
    path = Path(value)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = json.loads(value)
    if not isinstance(data, dict):
        raise TypeError("manual_frames must be a JSON object.")
    return {str(k): list(v) for k, v in data.items()}


def load_frame_records(path: Path) -> dict[str, FrameRecord]:
    records = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            record = FrameRecord(
                frame_id=row["frame_id"],
                label=int(row["label"]),
                baseline_score=float(row["baseline_score"]),
                candidate_score=float(row["candidate_score"]),
                delta_score=float(row["delta_score"]),
                baseline_rank=int(row["baseline_rank"]),
                candidate_rank=int(row["candidate_rank"]),
                delta_rank=int(row["delta_rank"]),
                baseline_contribution=float(row["baseline_contribution"]),
                candidate_contribution=float(row["candidate_contribution"]),
                delta_contribution=float(row["delta_contribution"]),
                positive_flip_count=int(row["positive_flip_count"]),
                negative_flip_count=int(row["negative_flip_count"]),
                net_flip_count=int(row["net_flip_count"]),
            )
            records[record.frame_id] = record
    return records


def load_pair_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_view_id(frame_id: str) -> int:
    match = re.search(r"_Cam(\d+)_", frame_id)
    if not match:
        raise ValueError(f"Cannot infer view id from frame_id: {frame_id}")
    return int(match.group(1))


def label_name_from_int(label: int) -> str:
    return "good" if label == 0 else "broken"


def resolve_modal_path(root: Path, modal: str, view_id: int, label_name: str, frame_id: str) -> Path:
    modal_root = "rail_mvtec" if modal == "rgb" else "rail_mvtec_depth"
    expected_ext = ".jpg" if modal == "rgb" else ".tiff"
    exact = root / modal_root / f"cam{view_id}" / "test" / label_name / f"{frame_id}{expected_ext}"
    if exact.exists():
        return exact
    parent = exact.parent
    matches = list(parent.glob(f"{frame_id}.*"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Cannot resolve {modal} path for {frame_id}: {exact}")


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if dst.exists():
        return
    if mode == "hardlink":
        os.link(src, dst)
    else:
        shutil.copy2(src, dst)


def load_depth_array(path: Path) -> np.ndarray:
    if tifffile is not None:
        array = tifffile.imread(path)
    else:  # pragma: no cover
        array = np.array(Image.open(path))
    if array.ndim >= 3:
        array = array[..., 0]
    return np.asarray(array, dtype=np.float32)


def build_depth_preview(depth_array: np.ndarray, pct_low: float, pct_high: float) -> tuple[Image.Image, float, float]:
    valid = np.isfinite(depth_array)
    if not np.any(valid):
        preview = np.zeros_like(depth_array, dtype=np.uint8)
        return Image.fromarray(preview, mode="L"), 0.0, 0.0

    values = depth_array[valid]
    low = float(np.percentile(values, pct_low))
    high = float(np.percentile(values, pct_high))
    if high <= low:
        low = float(values.min())
        high = float(values.max())
    if high <= low:
        high = low + 1.0
    scaled = np.clip((depth_array - low) / (high - low), 0.0, 1.0)
    preview = (scaled * 255.0).astype(np.uint8)
    return Image.fromarray(preview, mode="L"), low, high


def gather_comparison_names(rank_root: Path, requested: list[str] | None) -> list[str]:
    if requested:
        return requested
    names = []
    for child in sorted(rank_root.iterdir()):
        if child.is_dir() and (child / "frame_deltas.csv").exists():
            names.append(child.name)
    return names


def select_frame_ids(
    records: dict[str, FrameRecord],
    positive_pairs: list[dict[str, str]],
    negative_pairs: list[dict[str, str]],
    top_positive: int,
    top_negative: int,
    top_positive_pairs: int,
    top_negative_pairs: int,
    manual_frames: list[str],
) -> dict[str, list[str]]:
    reasons: dict[str, list[str]] = defaultdict(list)
    ranked = sorted(records.values(), key=lambda x: x.delta_contribution, reverse=True)

    for record in ranked[: max(0, top_positive)]:
        reasons[record.frame_id].append("top_positive_delta")

    negative_candidates = [row for row in sorted(records.values(), key=lambda x: x.delta_contribution) if row.delta_contribution < 0]
    for record in negative_candidates[: max(0, top_negative)]:
        reasons[record.frame_id].append("top_negative_delta")

    for row in positive_pairs[: max(0, top_positive_pairs)]:
        reasons[row["abnormal_frame_id"]].append("positive_pair_abnormal")
        reasons[row["normal_frame_id"]].append("positive_pair_normal")

    for row in negative_pairs[: max(0, top_negative_pairs)]:
        reasons[row["abnormal_frame_id"]].append("negative_pair_abnormal")
        reasons[row["normal_frame_id"]].append("negative_pair_normal")

    for frame_id in manual_frames:
        reasons[frame_id].append("manual_focus")

    return dict(reasons)


def build_pair_role_counts(
    positive_pairs: list[dict[str, str]],
    negative_pairs: list[dict[str, str]],
) -> dict[str, dict[str, int]]:
    counts = defaultdict(lambda: defaultdict(int))
    for row in positive_pairs:
        counts[row["abnormal_frame_id"]]["positive_as_abnormal"] += 1
        counts[row["normal_frame_id"]]["positive_as_normal"] += 1
    for row in negative_pairs:
        counts[row["abnormal_frame_id"]]["negative_as_abnormal"] += 1
        counts[row["normal_frame_id"]]["negative_as_normal"] += 1
    return counts


def export_comparison(
    rank_root: Path,
    test_root: Path,
    out_dir: Path,
    comparison: str,
    args,
    manual_frames_map: dict[str, list[str]],
) -> list[dict[str, object]]:
    comparison_root = rank_root / comparison
    frame_path = comparison_root / "frame_deltas.csv"
    positive_path = comparison_root / "pair_flips_positive.csv"
    negative_path = comparison_root / "pair_flips_negative.csv"
    if not frame_path.exists():
        raise FileNotFoundError(f"Missing frame deltas for {comparison}: {frame_path}")

    records = load_frame_records(frame_path)
    positive_pairs = load_pair_rows(positive_path)
    negative_pairs = load_pair_rows(negative_path)
    pair_role_counts = build_pair_role_counts(positive_pairs, negative_pairs)
    manual_frames = manual_frames_map.get(comparison, [])
    selected = select_frame_ids(
        records,
        positive_pairs,
        negative_pairs,
        args.top_positive,
        args.top_negative,
        args.top_positive_pairs,
        args.top_negative_pairs,
        manual_frames,
    )

    comparison_out = out_dir / comparison
    comparison_out.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    for frame_id in sorted(selected.keys()):
        if frame_id not in records:
            continue
        record = records[frame_id]
        label_name = label_name_from_int(record.label)
        view_id = parse_view_id(frame_id)
        rgb_src = resolve_modal_path(test_root, "rgb", view_id, label_name, frame_id)
        depth_src = resolve_modal_path(test_root, "depth", view_id, label_name, frame_id)

        frame_dir = comparison_out / f"{label_name}_{frame_id}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        rgb_dst = frame_dir / rgb_src.name
        depth_dst = frame_dir / depth_src.name
        depth_preview_dst = frame_dir / "depth_preview.png"

        link_or_copy(rgb_src, rgb_dst, args.copy_mode)
        link_or_copy(depth_src, depth_dst, args.copy_mode)

        depth_array = load_depth_array(depth_src)
        preview, low, high = build_depth_preview(depth_array, args.depth_pct_low, args.depth_pct_high)
        preview.save(depth_preview_dst)

        metadata = {
            "comparison": comparison,
            "frame_id": frame_id,
            "view_id": view_id,
            "label": record.label,
            "label_name": label_name,
            "selection_reasons": selected[frame_id],
            "rgb_source": str(rgb_src),
            "depth_source": str(depth_src),
            "rgb_export": str(rgb_dst),
            "depth_export": str(depth_dst),
            "depth_preview": str(depth_preview_dst),
            "baseline_score": record.baseline_score,
            "candidate_score": record.candidate_score,
            "delta_score": record.delta_score,
            "baseline_rank": record.baseline_rank,
            "candidate_rank": record.candidate_rank,
            "delta_rank": record.delta_rank,
            "baseline_contribution": record.baseline_contribution,
            "candidate_contribution": record.candidate_contribution,
            "delta_contribution": record.delta_contribution,
            "positive_flip_count": record.positive_flip_count,
            "negative_flip_count": record.negative_flip_count,
            "net_flip_count": record.net_flip_count,
            "positive_as_abnormal": pair_role_counts[frame_id].get("positive_as_abnormal", 0),
            "positive_as_normal": pair_role_counts[frame_id].get("positive_as_normal", 0),
            "negative_as_abnormal": pair_role_counts[frame_id].get("negative_as_abnormal", 0),
            "negative_as_normal": pair_role_counts[frame_id].get("negative_as_normal", 0),
            "depth_preview_pct_low": args.depth_pct_low,
            "depth_preview_pct_high": args.depth_pct_high,
            "depth_preview_value_low": low,
            "depth_preview_value_high": high,
        }
        with open(frame_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        manifest_rows.append(metadata)

    manifest_csv = comparison_out / "manifest.csv"
    manifest_fieldnames = [
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
        "positive_as_abnormal",
        "positive_as_normal",
        "negative_as_abnormal",
        "negative_as_normal",
        "rgb_source",
        "depth_source",
        "rgb_export",
        "depth_export",
        "depth_preview",
    ]
    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_fieldnames)
        writer.writeheader()
        for row in manifest_rows:
            csv_row = dict(row)
            csv_row["selection_reasons"] = "|".join(csv_row["selection_reasons"])
            writer.writerow({key: csv_row.get(key, "") for key in manifest_fieldnames})

    summary_txt = comparison_out / "summary.txt"
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(f"Comparison: {comparison}\n")
        f.write(f"Exported frames: {len(manifest_rows)}\n")
        for row in sorted(manifest_rows, key=lambda x: x["delta_contribution"], reverse=True):
            f.write(
                f"- {row['frame_id']} ({row['label_name']}): "
                f"delta_contribution={row['delta_contribution']:.8f}, "
                f"net_flip_count={row['net_flip_count']}, "
                f"reasons={','.join(row['selection_reasons'])}\n"
            )

    return manifest_rows


def main():
    args = build_parser().parse_args()
    rank_root = Path(args.rank_root)
    test_root = Path(args.test_root)
    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir(rank_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    manual_frames_map = load_manual_frames(args.manual_frames)
    comparisons = gather_comparison_names(rank_root, args.comparisons)

    all_rows: list[dict[str, object]] = []
    for comparison in comparisons:
        rows = export_comparison(rank_root, test_root, out_dir, comparison, args, manual_frames_map)
        all_rows.extend(rows)

    all_manifest_csv = out_dir / "manifest_all.csv"
    all_manifest_fieldnames = [
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
        "positive_as_abnormal",
        "positive_as_normal",
        "negative_as_abnormal",
        "negative_as_normal",
        "rgb_export",
        "depth_export",
        "depth_preview",
    ]
    with open(all_manifest_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_manifest_fieldnames)
        writer.writeheader()
        for row in all_rows:
            csv_row = dict(row)
            csv_row["selection_reasons"] = "|".join(csv_row["selection_reasons"])
            writer.writerow({key: csv_row.get(key, "") for key in all_manifest_fieldnames})

    summary_txt = out_dir / "summary.txt"
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("Key-frame export\n")
        f.write(f"created_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"rank_root: {rank_root}\n")
        f.write(f"test_root: {test_root}\n")
        f.write(f"comparisons: {', '.join(comparisons)}\n")
        f.write(f"copy_mode: {args.copy_mode}\n")
        f.write(
            f"selection: top_positive={args.top_positive}, top_negative={args.top_negative}, "
            f"top_positive_pairs={args.top_positive_pairs}, top_negative_pairs={args.top_negative_pairs}\n"
        )
        f.write(f"exported_frames_total: {len(all_rows)}\n\n")
        by_comparison = defaultdict(list)
        for row in all_rows:
            by_comparison[row["comparison"]].append(row)
        for comparison in comparisons:
            rows = sorted(by_comparison[comparison], key=lambda x: x["delta_contribution"], reverse=True)
            f.write(f"{comparison}: {len(rows)} frames\n")
            for row in rows:
                f.write(
                    f"  - {row['frame_id']} ({row['label_name']}): "
                    f"delta_contribution={row['delta_contribution']:.8f}, "
                    f"net_flip_count={row['net_flip_count']}, "
                    f"reasons={','.join(row['selection_reasons'])}\n"
                )
            f.write("\n")

    print(f"Key-frame output dir: {out_dir}")
    print(f"Key-frame manifest: {all_manifest_csv}")
    print(f"Key-frame summary: {summary_txt}")


if __name__ == "__main__":
    main()
