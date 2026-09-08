#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build compact paper-style qualitative figures from exported key-frame maps."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


DEFAULT_MAPS_ROOT = (
    "outputs/rail_peft/cam4_p1_20260501_225618/"
    "diagnostics/key_frame_maps_server_strict"
)
DEFAULT_OUT_DIR = (
    "outputs/rail_peft/cam4_p1_20260501_225618/"
    "diagnostics/paper_figures/p1_qualitative"
)


@dataclass(frozen=True)
class CaseSpec:
    group: str
    comparison: str
    case_dir: str
    short_name: str
    note: str


DEFAULT_CASES = [
    CaseSpec(
        group="cam4",
        comparison="cam4_depth",
        case_dir="good_20250417_123456_Cam4_00079",
        short_name="Cam4 normal",
        note="false positive suppressed",
    ),
    CaseSpec(
        group="cam4",
        comparison="cam4_depth",
        case_dir="broken_20251210_185619_Cam4_00024",
        short_name="Cam4 broken",
        note="ranking recovered",
    ),
    CaseSpec(
        group="cam4",
        comparison="cam4_depth",
        case_dir="good_20251210_185619_Cam4_00046",
        short_name="Cam4 normal",
        note="limitation case",
    ),
    CaseSpec(
        group="cam5",
        comparison="cam5_fusion",
        case_dir="broken_20251112_191827_Cam5_00067",
        short_name="Cam5 broken",
        note="top positive flip",
    ),
    CaseSpec(
        group="cam5",
        comparison="cam5_fusion",
        case_dir="broken_20251112_191827_Cam5_00123",
        short_name="Cam5 broken",
        note="subtle positive flip",
    ),
    CaseSpec(
        group="cam5",
        comparison="cam5_fusion",
        case_dir="broken_20251112_191827_Cam5_00007",
        short_name="Cam5 broken",
        note="negative flip",
    ),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create paper-ready P1 qualitative figures from key-frame map exports."
    )
    parser.add_argument("--maps_root", type=str, default=DEFAULT_MAPS_ROOT)
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--groups",
        nargs="*",
        default=["cam4", "cam5", "combined"],
        choices=["cam4", "cam5", "combined"],
        help="Which figures to write.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--panel_width", type=float, default=1.35)
    parser.add_argument("--row_height", type=float, default=3.7)
    parser.add_argument("--text_width", type=float, default=1.75)
    parser.add_argument(
        "--write_pdf",
        action="store_true",
        default=True,
        help="Also save PDF versions.",
    )
    parser.add_argument(
        "--no_pdf",
        action="store_false",
        dest="write_pdf",
    )
    return parser


def load_metadata(case_root: Path) -> dict:
    with (case_root / "metadata.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def merge_close_runs(runs: list[tuple[int, int]], max_gap: int = 70) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in runs:
        if not merged:
            merged.append((start, end))
            continue
        last_start, last_end = merged[-1]
        gap = start - last_end
        combined_width = end - last_start
        if gap <= max_gap and combined_width <= 720:
            merged[-1] = (last_start, end)
        else:
            merged.append((start, end))
    return merged


def find_panel_runs(image: np.ndarray) -> list[tuple[int, int]]:
    """Find the six image panel x-ranges in the exported 6-panel figure."""
    mask = (image < 245).any(axis=2)
    col_frac = mask[150:, :].mean(axis=0)
    active = col_frac > 0.2
    runs: list[tuple[int, int]] = []
    in_run = False
    start = 0
    for idx, value in enumerate(active):
        if value and not in_run:
            start = idx
            in_run = True
        if in_run and (not value or idx == len(active) - 1):
            end = idx if not value else idx + 1
            if end - start > 100:
                runs.append((start, end))
            in_run = False
    runs = merge_close_runs(runs)
    if len(runs) == 7:
        # Depth-preview panels can be split by a bright center band. Merge the
        # adjacent pair whose combined width best matches the other panel widths.
        widths = [end - start for start, end in runs]
        target_width = float(np.median([w for w in widths if w > 500] or widths))
        best_idx = None
        best_score = float("inf")
        for idx in range(len(runs) - 1):
            start, _ = runs[idx]
            _, end = runs[idx + 1]
            gap = runs[idx + 1][0] - runs[idx][1]
            combined_width = end - start
            if gap > 220 or combined_width > 760:
                continue
            score = abs(combined_width - target_width)
            if score < best_score:
                best_idx = idx
                best_score = score
        if best_idx is not None:
            runs = (
                runs[:best_idx]
                + [(runs[best_idx][0], runs[best_idx + 1][1])]
                + runs[best_idx + 2 :]
            )
    if len(runs) != 6:
        raise RuntimeError(f"Expected 6 panel runs, got {len(runs)}: {runs}")
    return runs


def crop_selected_panels(figure_path: Path) -> dict[str, Image.Image]:
    image = np.asarray(Image.open(figure_path).convert("RGB"))
    runs = find_panel_runs(image)
    height = image.shape[0]
    y0 = 160
    y1 = height - 18
    selected = {
        "input": runs[0],
        "baseline": runs[3],
        "p1": runs[4],
        "delta": runs[5],
    }
    crops: dict[str, Image.Image] = {}
    pil_image = Image.fromarray(image)
    for name, (x0, x1) in selected.items():
        crops[name] = pil_image.crop((x0, y0, x1, y1))
    return crops


def format_case_text(spec: CaseSpec, meta: dict) -> str:
    frame = meta["frame_id"].split("_")[-1]
    before = float(meta["baseline_score"])
    after = float(meta["candidate_score"])
    net = int(meta["net_flip_count"])
    source = meta["source"]
    return (
        f"{spec.short_name}\n"
        f"{meta['label_name']} #{frame}\n"
        f"{spec.note}\n"
        f"{source}: {before:.4f}->{after:.4f}\n"
        f"rank {meta['baseline_rank']}->{meta['candidate_rank']} | flips {net:+d}"
    )


def draw_group(
    specs: list[CaseSpec],
    maps_root: Path,
    out_path: Path,
    title: str,
    dpi: int,
    panel_width: float,
    row_height: float,
    text_width: float,
    write_pdf: bool,
) -> list[dict]:
    columns = ["Case", "Input", "Baseline", "P1", "Delta"]
    width_ratios = [text_width, panel_width, panel_width, panel_width, panel_width]
    fig_width = sum(width_ratios)
    fig_height = row_height * len(specs) + 0.55
    fig, axes = plt.subplots(
        nrows=len(specs),
        ncols=len(columns),
        figsize=(fig_width, fig_height),
        gridspec_kw={"width_ratios": width_ratios, "wspace": 0.045, "hspace": 0.065},
    )
    if len(specs) == 1:
        axes = np.asarray([axes])
    fig.suptitle(title, fontsize=10, y=0.992)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.965, bottom=0.01)

    rows_for_csv: list[dict] = []
    for row_idx, spec in enumerate(specs):
        case_root = maps_root / spec.comparison / spec.case_dir
        meta = load_metadata(case_root)
        crops = crop_selected_panels(case_root / "figure.png")
        rows_for_csv.append(
            {
                "figure": out_path.name,
                "group": spec.group,
                "comparison": spec.comparison,
                "case_dir": spec.case_dir,
                "frame_id": meta["frame_id"],
                "label_name": meta["label_name"],
                "source": meta["source"],
                "baseline_score": meta["baseline_score"],
                "candidate_score": meta["candidate_score"],
                "delta_score": meta["delta_score"],
                "baseline_rank": meta["baseline_rank"],
                "candidate_rank": meta["candidate_rank"],
                "net_flip_count": meta["net_flip_count"],
                "delta_contribution": meta["delta_contribution"],
                "note": spec.note,
            }
        )

        for col_idx in range(len(columns)):
            ax = axes[row_idx, col_idx]
            ax.set_axis_off()
            if row_idx == 0:
                ax.set_title(columns[col_idx], fontsize=8, pad=5)
            if col_idx == 0:
                ax.text(
                    0.0,
                    0.5,
                    format_case_text(spec, meta),
                    va="center",
                    ha="left",
                    fontsize=7.4,
                    linespacing=1.25,
                    transform=ax.transAxes,
                )
            else:
                key = columns[col_idx].lower()
                if key == "p1":
                    key = "p1"
                ax.imshow(crops[key])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    if write_pdf:
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return rows_for_csv


def main() -> None:
    args = build_parser().parse_args()
    maps_root = Path(args.maps_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    group_to_specs = {
        "cam4": [spec for spec in DEFAULT_CASES if spec.group == "cam4"],
        "cam5": [spec for spec in DEFAULT_CASES if spec.group == "cam5"],
        "combined": DEFAULT_CASES,
    }
    titles = {
        "cam4": "P1 qualitative cases on Cam4 Depth branch",
        "cam5": "Cam5 diagnostic fusion cases with Cam4 P1 attached",
        "combined": "P1 qualitative summary: Cam4 repair and Cam5 subtle fusion shifts",
    }
    output_names = {
        "cam4": "p1_qualitative_cam4_depth.png",
        "cam5": "p1_qualitative_cam5_fusion.png",
        "combined": "p1_qualitative_combined.png",
    }

    all_rows: list[dict] = []
    written: list[Path] = []
    for group in args.groups:
        out_path = out_dir / output_names[group]
        rows = draw_group(
            specs=group_to_specs[group],
            maps_root=maps_root,
            out_path=out_path,
            title=titles[group],
            dpi=args.dpi,
            panel_width=args.panel_width,
            row_height=args.row_height,
            text_width=args.text_width,
            write_pdf=args.write_pdf,
        )
        all_rows.extend(rows)
        written.append(out_path)
        if args.write_pdf:
            written.append(out_path.with_suffix(".pdf"))

    manifest_path = out_dir / "selected_cases.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    readme_path = out_dir / "README.txt"
    with readme_path.open("w", encoding="utf-8") as f:
        f.write("P1 qualitative figures\n")
        f.write(f"maps_root: {maps_root}\n")
        f.write("columns: Case, Input, Baseline, P1, Delta\n")
        f.write("figures:\n")
        for path in written:
            f.write(f"  - {path}\n")
        f.write(f"selected_cases: {manifest_path}\n")

    print(f"Wrote {len(written)} figure files")
    for path in written:
        print(path)
    print(manifest_path)
    print(readme_path)


if __name__ == "__main__":
    main()
