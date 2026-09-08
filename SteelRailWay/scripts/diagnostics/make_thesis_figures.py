#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate paper figures for the rail PEFT/CAD thesis.

The script is intentionally data-light: it uses numeric results already
reported in the experiment JSON/CSV files and re-composes exported qualitative
panels into publication-ready figures.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "figures"
QUAL_ROOT = (
    ROOT
    / "outputs"
    / "rail_peft"
    / "cam4_p1_20260501_225618"
    / "diagnostics"
    / "key_frame_maps_server_strict"
)
CFCA_QUAL_ROOT = ROOT / "outputs" / "thesis_figures" / "cam4_cfca_qualitative" / "maps"


def save_all(fig, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{stem}.svg", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def draw_mismatch_phase() -> None:
    points = [
        {
            "name": "peft_full",
            "short": "P1 only",
            "x": -0.2125,
            "y": 0.0375,
            "rgb_iso": 0.8125,
            "depth_iso": 0.6375,
            "fusion": 0.6875,
            "mode": "Lagged mismatch",
            "color": "#dc2626",
            "marker": "o",
            "size": 90,
            "offset": (-0.075, 0.060),
        },
        {
            "name": "no_cf_ca",
            "short": "No CF/CA",
            "x": 0.0,
            "y": 0.0,
            "rgb_iso": 0.8625,
            "depth_iso": 0.7125,
            "fusion": 0.7375,
            "mode": "Reference",
            "color": "#64748b",
            "marker": "o",
            "size": 80,
            "offset": (0.030, 0.030),
        },
        {
            "name": "cf_ca_repair",
            "short": "CF/CA repair",
            "x": 0.0,
            "y": 0.1375,
            "rgb_iso": 0.6750,
            "depth_iso": 0.6625,
            "fusion": 0.7750,
            "mode": "Healthy synergy",
            "color": "#16a34a",
            "marker": "*",
            "size": 260,
            "offset": (0.035, 0.025),
        },
        {
            "name": "peft_full_then_cf_ca",
            "short": "Continue",
            "x": 0.3250,
            "y": 0.2500,
            "rgb_iso": 0.3625,
            "depth_iso": 0.5875,
            "fusion": 0.6500,
            "mode": "Over-coupling",
            "color": "#7c3aed",
            "marker": "o",
            "size": 95,
            "offset": (-0.205, 0.030),
        },
    ]

    by_name = {p["name"]: p for p in points}
    fig, (ax, ax_iso) = plt.subplots(
        1,
        2,
        figsize=(12.0, 4.8),
        gridspec_kw={"width_ratios": [1.16, 1.0]},
    )
    fig.suptitle(
        "Isolated AUROC diagnostic view for Cam4",
        fontsize=12.5,
        weight="bold",
        y=0.985,
    )

    ax.set_xlim(-0.28, 0.35)
    ax.set_ylim(-0.04, 0.29)
    ax.set_title("(a) Cross-modal contribution plane", loc="left", fontsize=11, weight="bold")
    ax.add_patch(Rectangle((-0.28, -0.04), 0.28, 0.33, color="#fee2e2", alpha=0.38, zorder=0))
    ax.add_patch(Rectangle((-0.018, 0.085), 0.10, 0.11, color="#dcfce7", alpha=0.58, zorder=0))
    ax.add_patch(Rectangle((0.18, 0.13), 0.17, 0.16, color="#ede9fe", alpha=0.62, zorder=0))
    ax.axhline(0, color="#475569", linewidth=1.0)
    ax.axvline(0, color="#475569", linewidth=1.0)
    ax.grid(True, color="#cbd5e1", linewidth=0.8, alpha=0.75)

    ax.text(-0.265, 0.265, "RGB harmed\n($\\Delta RGB<0$)", color="#991b1b", fontsize=9, weight="bold")
    ax.text(0.012, 0.172, "Healthy\nsynergy", color="#166534", fontsize=9, weight="bold")
    ax.text(0.205, 0.265, "Check isolated\ncapacity", color="#5b21b6", fontsize=9, weight="bold")

    def draw_route(src: str, dst: str, color: str, label: str, text_xy: tuple[float, float], rad: float) -> None:
        p0 = by_name[src]
        p1 = by_name[dst]
        ax.annotate(
            "",
            xy=(p1["x"], p1["y"]),
            xytext=(p0["x"], p0["y"]),
            arrowprops=dict(
                arrowstyle="->",
                color=color,
                lw=2.0,
                shrinkA=8,
                shrinkB=9,
                connectionstyle=f"arc3,rad={rad}",
            ),
            zorder=2,
        )
        ax.text(
            text_xy[0],
            text_xy[1],
            label,
            fontsize=8.5,
            color=color,
            weight="bold",
            bbox=dict(facecolor="white", edgecolor=color, boxstyle="round,pad=0.25", alpha=0.92),
        )

    draw_route("peft_full", "cf_ca_repair", "#16a34a", "targeted\nCF/CA repair", (-0.12, 0.125), 0.18)
    draw_route("peft_full", "peft_full_then_cf_ca", "#7c3aed", "continue\njoint tuning", (0.085, 0.230), -0.11)

    for p in points:
        ax.scatter(
            p["x"],
            p["y"],
            s=p["size"],
            color=p["color"],
            marker=p["marker"],
            edgecolor="white",
            linewidth=1.3,
            zorder=3,
            label=p["mode"],
        )
        dx, dy = p["offset"]
        if p["name"] == "peft_full":
            note = "P1 only\nRGB cross is polluted"
        elif p["name"] == "cf_ca_repair":
            note = "CF/CA repair\nDepth helped, RGB not harmed"
        elif p["name"] == "peft_full_then_cf_ca":
            note = "Continuation\npositive $\\Delta$ hides iso collapse"
        else:
            note = "No CF/CA\nreference"
        ax.annotate(
            note,
            xy=(p["x"], p["y"]),
            xytext=(p["x"] + dx, p["y"] + dy),
            arrowprops=dict(arrowstyle="-", color=p["color"], lw=1.0),
            fontsize=9,
            color="#0f172a",
            ha="left",
            va="center",
            bbox=dict(facecolor="white", edgecolor=p["color"], boxstyle="round,pad=0.25", alpha=0.93),
        )

    ax.set_xlabel(r"$\Delta RGB = AUROC_{cross} - AUROC_{isolated}$", fontsize=11)
    ax.set_ylabel(r"$\Delta Depth = AUROC_{cross} - AUROC_{isolated}$", fontsize=11)

    ax_iso.set_title("(b) Isolated capacity check", loc="left", fontsize=11, weight="bold")
    labels_short = [p["short"] for p in points]
    x = np.arange(len(points))
    width = 0.30
    rgb_iso = np.array([p["rgb_iso"] for p in points], dtype=float)
    depth_iso = np.array([p["depth_iso"] for p in points], dtype=float)
    fusion = np.array([p["fusion"] for p in points], dtype=float)
    ax_iso.axhspan(0.30, 0.60, color="#fee2e2", alpha=0.23, zorder=0)
    ax_iso.bar(x - width / 2, rgb_iso, width, color="#38bdf8", edgecolor="#0369a1", label="RGB isolated")
    ax_iso.bar(x + width / 2, depth_iso, width, color="#fbbf24", edgecolor="#b45309", label="Depth isolated")
    ax_iso.plot(x, fusion, color="#111827", marker="D", markersize=5.5, linewidth=1.5, label="Fusion AUROC")
    ax_iso.axhline(0.50, color="#ef4444", linestyle="--", linewidth=1.0)
    ax_iso.text(3.35, 0.508, "0.5 random", color="#ef4444", fontsize=8, va="bottom", ha="right")

    for idx, value in enumerate(rgb_iso):
        ax_iso.text(idx - width / 2, value + 0.012, f"{value:.3f}", ha="center", va="bottom", fontsize=7.5)
    for idx, value in enumerate(depth_iso):
        ax_iso.text(idx + width / 2, value + 0.012, f"{value:.3f}", ha="center", va="bottom", fontsize=7.5)
    for idx, value in enumerate(fusion):
        ax_iso.text(idx, value - 0.038, f"{value:.3f}", ha="center", va="top", fontsize=7.5, color="#111827")

    ax_iso.annotate(
        "over-coupling:\nRGB isolated collapses",
        xy=(3 - width / 2, rgb_iso[3]),
        xytext=(2.20, 0.435),
        arrowprops=dict(arrowstyle="->", color="#7c3aed", lw=1.4),
        fontsize=8.5,
        color="#5b21b6",
        weight="bold",
        bbox=dict(facecolor="white", edgecolor="#7c3aed", boxstyle="round,pad=0.25", alpha=0.94),
    )
    ax_iso.annotate(
        "best fusion\nwithout iso collapse",
        xy=(2, fusion[2]),
        xytext=(1.36, 0.825),
        arrowprops=dict(arrowstyle="->", color="#16a34a", lw=1.4),
        fontsize=8.5,
        color="#166534",
        weight="bold",
        bbox=dict(facecolor="white", edgecolor="#16a34a", boxstyle="round,pad=0.25", alpha=0.94),
    )
    ax_iso.set_xticks(x)
    ax_iso.set_xticklabels(labels_short, fontsize=9)
    ax_iso.set_ylim(0.30, 0.90)
    ax_iso.set_ylabel("AUROC", fontsize=11)
    ax_iso.grid(axis="y", color="#cbd5e1", linewidth=0.8, alpha=0.75)
    ax_iso.legend(loc="lower left", fontsize=8.5, frameon=True, framealpha=0.95)

    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=2.0)
    save_all(fig, "fig4_1_mismatch_phase")


def draw_forgetting_bar() -> None:
    cameras = ["Cam1", "Cam4", "Cam5"]
    baseline = np.array([0.7364, 0.3500, 0.7551])
    t1 = np.array([0.7364, 0.7750, 0.8367])
    delta = t1 - baseline
    labels = ["+0.0000\nzero degradation", "+0.4250\nnew-task repair", "+0.0816\npositive transfer"]

    x = np.arange(len(cameras))
    width = 0.32
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.bar(x - width / 2, baseline, width, label=r"$T_0$ baseline", color="#bfdbfe", edgecolor="#1d4ed8")
    ax.bar(x + width / 2, t1, width, label=r"$T_1$ + PEFT", color="#2563eb", edgecolor="#1e40af")
    ax.axhline(0.5, color="#ef4444", linestyle="--", linewidth=1.2, label="0.5 random baseline")

    for idx, text in enumerate(labels):
        color = "#166534" if idx != 1 else "#b91c1c"
        ax.text(
            x[idx],
            max(baseline[idx], t1[idx]) + 0.035,
            text,
            ha="center",
            va="bottom",
            fontsize=9,
            color=color,
            weight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(cameras, fontsize=11)
    ax.set_ylim(0.0, 0.95)
    ax.set_ylabel("Fusion AUROC", fontsize=11)
    ax.set_title("Cross-view Anti-forgetting Evaluation", fontsize=15, weight="bold", pad=12)
    ax.text(0.98, 0.08, "BWT = +0.041, FWT = +0.4250", transform=ax.transAxes, ha="right", fontsize=10)
    ax.grid(axis="y", color="#cbd5e1", linewidth=0.8, alpha=0.8)
    ax.legend(loc="lower right", frameon=True, framealpha=0.96)
    save_all(fig, "fig4_2_cross_view_forgetting")


def find_panel_runs(image: np.ndarray) -> list[tuple[int, int]]:
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

    merged: list[tuple[int, int]] = []
    for start, end in runs:
        if not merged:
            merged.append((start, end))
            continue
        last_start, last_end = merged[-1]
        if start - last_end <= 70 and end - last_start <= 720:
            merged[-1] = (last_start, end)
        else:
            merged.append((start, end))
    if len(merged) == 7:
        widths = [b - a for a, b in merged]
        target = float(np.median([w for w in widths if w > 500] or widths))
        best_idx = None
        best_score = float("inf")
        for idx in range(len(merged) - 1):
            start, _ = merged[idx]
            _, end = merged[idx + 1]
            gap = merged[idx + 1][0] - merged[idx][1]
            combined = end - start
            if gap > 220 or combined > 760:
                continue
            score = abs(combined - target)
            if score < best_score:
                best_idx = idx
                best_score = score
        if best_idx is not None:
            merged = (
                merged[:best_idx]
                + [(merged[best_idx][0], merged[best_idx + 1][1])]
                + merged[best_idx + 2 :]
            )
    if len(merged) != 6:
        raise RuntimeError(f"Expected six panels in exported figure, got {len(merged)}: {merged}")
    return merged


def crop_exported_panels(case_dir: Path) -> dict[str, Image.Image]:
    image = np.asarray(Image.open(case_dir / "figure.png").convert("RGB"))
    runs = find_panel_runs(image)
    y0, y1 = 160, image.shape[0] - 18
    names = {
        "input": runs[0],
        "baseline": runs[3],
        "peft": runs[4],
        "delta": runs[5],
    }
    pil = Image.fromarray(image)
    return {name: pil.crop((x0, y0, x1, y1)) for name, (x0, x1) in names.items()}


def load_selected_case_rows() -> dict[str, dict[str, str]]:
    csv_path = (
        ROOT
        / "outputs"
        / "rail_peft"
        / "cam4_p1_20260501_225618"
        / "diagnostics"
        / "paper_figures"
        / "p1_qualitative"
        / "selected_cases.csv"
    )
    rows = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["figure"] == "p1_qualitative_cam4_depth.png":
                rows[row["case_dir"]] = row
    return rows


def draw_qualitative_grid() -> None:
    cases = [
        ("Normal sample", "good_20250417_123456_Cam4_00079", "false positive suppressed"),
        ("Defect sample", "broken_20251210_185619_Cam4_00024", "defect response recovered"),
        ("Failure sample", "good_20251210_185619_Cam4_00046", "remaining local false alarm"),
    ]
    rows = load_selected_case_rows()
    cols = ["RGB input", "Baseline map", "DepthAffinePEFT", "+ CF/CA SelectFT"]
    fig, axes = plt.subplots(
        nrows=3,
        ncols=4,
        figsize=(9.2, 10.4),
        gridspec_kw={"wspace": 0.035, "hspace": 0.11},
    )
    fig.suptitle("Qualitative Heatmap Comparison on Cam4", fontsize=15, weight="bold", y=0.992)

    for row_idx, (row_label, case_name, note) in enumerate(cases):
        panels = crop_exported_panels(QUAL_ROOT / "cam4_depth" / case_name)
        cfca_panels = crop_exported_panels(CFCA_QUAL_ROOT / "cam4_fusion_cfca" / case_name)
        meta = rows.get(case_name, {})
        for col_idx, col in enumerate(cols):
            ax = axes[row_idx, col_idx]
            ax.axis("off")
            if row_idx == 0:
                ax.set_title(col, fontsize=10, pad=5)
            if col_idx == 3:
                ax.imshow(cfca_panels["peft"], aspect="auto")
            else:
                key = ["input", "baseline", "peft"][col_idx]
                ax.imshow(panels[key], aspect="auto")
            if col_idx == 0:
                text = row_label
                if meta:
                    text += (
                        f"\n{meta.get('frame_id', '').split('_')[-1]}"
                        f"\n{note}"
                    )
                ax.text(
                    0.02,
                    0.98,
                    text,
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=8,
                    color="white",
                    bbox=dict(facecolor="#0f172a", alpha=0.78, edgecolor="none", pad=3),
                )
            if col_idx in (1, 2) and meta:
                score = meta["baseline_score"] if col_idx == 1 else meta["candidate_score"]
                ax.text(
                    0.98,
                    0.98,
                    f"score={float(score):.4f}",
                    transform=ax.transAxes,
                    va="top",
                    ha="right",
                    fontsize=8,
                    color="white",
                    bbox=dict(facecolor="#0f172a", alpha=0.70, edgecolor="none", pad=3),
                )
            if col_idx == 3:
                repair_meta = CFCA_QUAL_ROOT / "cam4_fusion_cfca" / case_name / "metadata.json"
                if repair_meta.exists():
                    import json

                    payload = json.loads(repair_meta.read_text(encoding="utf-8"))
                    ax.text(
                        0.98,
                        0.98,
                        f"score={float(payload['candidate_score']):.4f}",
                        transform=ax.transAxes,
                        va="top",
                        ha="right",
                        fontsize=8,
                        color="white",
                        bbox=dict(facecolor="#0f172a", alpha=0.70, edgecolor="none", pad=3),
                    )

    save_all(fig, "fig4_3_qualitative_heatmaps")


def draw_method_overview() -> None:
    """Figure 3-1: Overall method framework diagram."""
    from matplotlib.patches import FancyBboxPatch

    plt.rcParams["font.family"] = "Microsoft YaHei"

    fig, ax = plt.subplots(figsize=(15.5, 7.0))
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 7.0)
    ax.axis("off")

    # ── Color palette ──
    FROZEN_FACE = "#ecfdf5"
    FROZEN_EDGE = "#059669"
    FROZEN_SUB = "#d1fae5"
    TRAIN_FACE = "#eff6ff"
    TRAIN_EDGE = "#2563eb"
    INPUT_FACE = "#f8fafc"
    INPUT_EDGE = "#475569"
    OUTPUT_FACE = "#fef3c7"
    OUTPUT_EDGE = "#d97706"
    DIAG_FACE = "#fdf2f8"
    DIAG_EDGE = "#db2777"
    PARAM_FACE = "#f5f3ff"
    PARAM_EDGE = "#7c3aed"
    ARROW_COLOR = "#334155"

    def draw_box(x, y, w, h, text, face, edge, fontsize=9, weight="normal",
                 text_color="#0f172a", lw=1.8, ls="-"):
        rect = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.12",
            facecolor=face, edgecolor=edge, linewidth=lw, zorder=2, linestyle=ls,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, weight=weight, color=text_color, zorder=3)

    def draw_arrow(x0, y0, x1, y1, color=ARROW_COLOR, lw=1.5):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                                    connectionstyle="arc3,rad=0"), zorder=1)

    def draw_small_label(x, y, text, color="#0f172a", fontsize=7.5):
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
                color=color, zorder=4)

    # ═══════════════════════════════════════════
    # Title
    # ═══════════════════════════════════════════
    ax.text(7.75, 6.70, "图 3-1  整体方法框架图", ha="center", va="center",
            fontsize=15, weight="bold", color="#0f172a")

    # ═══════════════════════════════════════════
    # ① Inputs (left)
    # ═══════════════════════════════════════════
    draw_box(0.3, 4.3, 2.0, 0.95, "Intensity\n(RGB Image)", INPUT_FACE, INPUT_EDGE)
    draw_box(0.3, 2.5, 2.0, 0.95, "Depth\n(16-bit TIFF)", INPUT_FACE, INPUT_EDGE)
    ax.text(1.3, 5.55, "① 双模态输入", ha="center", fontsize=8.5, color="#64748b", weight="bold")

    # ═══════════════════════════════════════════
    # ② DepthAffinePEFT (trainable, Depth path only)
    # ═══════════════════════════════════════════
    draw_box(3.2, 2.5, 2.6, 0.95,
             "DepthAffinePEFT\n$x_d' = \\gamma \\, x_d + \\beta$",
             TRAIN_FACE, TRAIN_EDGE, weight="bold")
    draw_small_label(4.5, 3.60, "可训练 (2 params)", "#2563eb")
    ax.text(4.5, 5.55, "② 输入端修正", ha="center", fontsize=8.5, color="#64748b", weight="bold")

    # Arrows: inputs → next
    draw_arrow(2.3, 4.78, 7.0, 4.78)   # Intensity → backbone
    draw_arrow(2.3, 2.98, 3.2, 2.98)    # Depth → PEFT
    draw_arrow(5.8, 2.98, 7.0, 3.60)    # PEFT → backbone

    # ═══════════════════════════════════════════
    # ③ TRD Backbone (center, frozen enclosure)
    # ═══════════════════════════════════════════
    bb_x, bb_y, bb_w, bb_h = 7.0, 2.2, 4.4, 3.5
    ax.add_patch(FancyBboxPatch(
        (bb_x, bb_y), bb_w, bb_h, boxstyle="round,pad=0.18",
        facecolor=FROZEN_FACE, edgecolor=FROZEN_EDGE, linewidth=2.2, zorder=2))
    ax.text(bb_x + bb_w / 2, bb_y + bb_h - 0.30, "Frozen TRD Backbone",
            ha="center", va="center", fontsize=11.5, weight="bold", color="#047857", zorder=3)
    ax.text(bb_x + bb_w / 2, bb_y + bb_h - 0.65, "(Teacher-Student 蒸馏 + 跨模态融合)",
            ha="center", va="center", fontsize=8, color="#047857", zorder=3)

    # Sub-modules inside backbone: 2×2 grid
    sub_w, sub_h = 1.8, 0.85
    sub_x0, sub_y0 = bb_x + 0.40, bb_y + 0.45
    gap_x, gap_y = 0.40, 0.55
    # Row 1: Teacher Encoder (frozen) + Student Decoder (frozen)
    draw_box(sub_x0, sub_y0 + sub_h + gap_y, sub_w, sub_h,
             "Teacher Encoder\n(WRN50_2, frozen)", FROZEN_SUB, FROZEN_EDGE, fontsize=7.2, lw=1.3)
    draw_box(sub_x0 + sub_w + gap_x, sub_y0 + sub_h + gap_y, sub_w, sub_h,
             "Student Decoder\n(frozen)", FROZEN_SUB, FROZEN_EDGE, fontsize=7.2, lw=1.3)
    # Row 2: CF Module + CA Module (partially unfrozen → dual-color border)
    # Use a split visual: frozen fill but train-like edge
    CF_PARTIAL_FACE = "#e0f2fe"  # pale blue-green mix hint
    CA_PARTIAL_FACE = "#e0f2fe"
    CF_PARTIAL_EDGE = "#0284c7"  # sky blue border = trainable subset
    CA_PARTIAL_EDGE = "#0284c7"
    draw_box(sub_x0, sub_y0, sub_w, sub_h,
             "CF Module\n(Cross-Feature, SelFT)", CF_PARTIAL_FACE, CF_PARTIAL_EDGE,
             fontsize=7.2, lw=1.8, weight="bold", text_color="#0369a1")
    draw_box(sub_x0 + sub_w + gap_x, sub_y0, sub_w, sub_h,
             "CA Module\n(Cross-Attention, SelFT)", CA_PARTIAL_FACE, CA_PARTIAL_EDGE,
             fontsize=7.2, lw=1.8, weight="bold", text_color="#0369a1")

    # Small arrows inside backbone: Teacher → Student ↓ CF/CA
    ax.annotate("", xy=(sub_x0 + sub_w + gap_x, sub_y0 + sub_h + gap_y + sub_h / 2),
                xytext=(sub_x0 + sub_w + 0.08, sub_y0 + sub_h + gap_y + sub_h / 2),
                arrowprops=dict(arrowstyle="->", color="#94a3b8", lw=0.8), zorder=1)

    draw_small_label(bb_x + bb_w / 2, bb_y - 0.10,
                     "除 CF/CA 选择性微调参数外全部冻结", "#059669")

    # CF/CA selective FT annotation
    ax.annotate("CF/CA Selective FT\n(~108 params unfrozen)",
                xy=(sub_x0 + sub_w / 2, sub_y0 - 0.25),
                xytext=(sub_x0 + sub_w / 2, sub_y0 - 0.70),
                ha="center", va="top", fontsize=7.2, color="#0369a1", weight="bold",
                arrowprops=dict(arrowstyle="->", color="#0284c7", lw=1.0), zorder=5)

    ax.text(9.2, 5.55, "③ 主干推理 + 协同修复", ha="center", fontsize=8.5, color="#64748b", weight="bold")

    # ═══════════════════════════════════════════
    # ④ Output: Fusion Anomaly Map
    # ═══════════════════════════════════════════
    draw_box(12.5, 3.4, 2.7, 1.1,
             "Fusion Anomaly Map\n$M = M_i^{norm} + M_d^{norm}$",
             OUTPUT_FACE, OUTPUT_EDGE, weight="bold")
    draw_arrow(11.4, 3.95, 12.5, 3.95)
    ax.text(13.85, 5.55, "④ 异常检测输出", ha="center", fontsize=8.5, color="#64748b", weight="bold")

    # ═══════════════════════════════════════════
    # Legend
    # ═══════════════════════════════════════════
    lx, ly = 0.3, 5.9
    ax.add_patch(FancyBboxPatch((lx, ly), 0.38, 0.28,
        boxstyle="round,pad=0.04", facecolor=FROZEN_FACE, edgecolor=FROZEN_EDGE, linewidth=1.3))
    ax.text(lx + 0.80, ly + 0.14, "冻结模块", fontsize=8, va="center", color="#047857")
    ax.add_patch(FancyBboxPatch((lx + 2.6, ly), 0.38, 0.28,
        boxstyle="round,pad=0.04", facecolor=TRAIN_FACE, edgecolor=TRAIN_EDGE, linewidth=1.3))
    ax.text(lx + 2.6 + 0.80, ly + 0.14, "可训练模块", fontsize=8, va="center", color="#2563eb")
    ax.add_patch(FancyBboxPatch((lx + 5.2, ly), 0.38, 0.28,
        boxstyle="round,pad=0.04", facecolor=CF_PARTIAL_FACE, edgecolor=CF_PARTIAL_EDGE, linewidth=1.3))
    ax.text(lx + 5.2 + 0.80, ly + 0.14, "选择性微调 (部分参数可训练)", fontsize=8, va="center", color="#0369a1")

    # ═══════════════════════════════════════════
    # Isolated AUROC diagnostic bar (top)
    # ═══════════════════════════════════════════
    draw_box(3.2, 6.1, 9.5, 0.52,
             "isolated AUROC 诊断模块:  $\\Delta = AUROC_{cross} - AUROC_{isolated}$  →  量化跨模态路径贡献度",
             DIAG_FACE, DIAG_EDGE, fontsize=9.5, weight="bold", text_color="#9d174d", lw=1.5)
    ax.annotate("", xy=(7.9, 5.7), xytext=(7.9, 6.1),
                arrowprops=dict(arrowstyle="<->", color="#db2777", lw=1.2), zorder=1)
    draw_small_label(8.5, 5.88, "评估入口", "#db2777")

    # ═══════════════════════════════════════════
    # Parameter budget box (right side)
    # ═══════════════════════════════════════════
    param_str = ("训练参数统计\n"
                 "  • DepthAffinePEFT: γ, β  (2 个)\n"
                 "  • CF/CA Selective FT: ≈ 108 个\n"
                 "  • 主干参数: 0 (全部冻结)\n"
                 "  ─────────────────────\n"
                 "  总计可训练: ≈ 110 个")
    ax.text(12.5, 5.95, param_str, ha="left", va="top", fontsize=8.0,
            color="#4c1d95",
            bbox=dict(facecolor=PARAM_FACE, edgecolor=PARAM_EDGE,
                      boxstyle="round,pad=0.5", linewidth=1.5), zorder=3)

    # ═══════════════════════════════════════════
    # CAD framework annotation (bottom)
    # ═══════════════════════════════════════════
    ax.text(7.75, 0.55,
            "CAD 补丁式持续适配: 每个任务 $\\mathcal{T}_k$ 生成独立补丁 "
            "$P_k = \\{\\phi_{affine}^k,\\ \\phi_{cf/ca}^k,\\ m_k\\}$  "
            "→ 补丁仓库管理, 可独立加载/卸载/回滚",
            ha="center", va="center", fontsize=9.0, style="italic", color="#475569", zorder=3)

    # ═══════════════════════════════════════════
    # Flow summary arrow (bottom)
    # ═══════════════════════════════════════════
    flow_y = 1.30
    ax.text(0.3, flow_y, "输入端修正", ha="left", fontsize=8, color="#2563eb", weight="bold")
    ax.annotate("", xy=(2.8, flow_y), xytext=(1.7, flow_y),
                arrowprops=dict(arrowstyle="->", color="#94a3b8", lw=1.0))
    ax.text(3.0, flow_y, "冻结主干推理", ha="left", fontsize=8, color="#059669", weight="bold")
    ax.annotate("", xy=(5.8, flow_y), xytext=(4.4, flow_y),
                arrowprops=dict(arrowstyle="->", color="#94a3b8", lw=1.0))
    ax.text(6.0, flow_y, "协同选择性微调", ha="left", fontsize=8, color="#0369a1", weight="bold")
    ax.annotate("", xy=(8.8, flow_y), xytext=(7.5, flow_y),
                arrowprops=dict(arrowstyle="->", color="#94a3b8", lw=1.0))
    ax.text(9.0, flow_y, "融合异常图", ha="left", fontsize=8, color="#d97706", weight="bold")
    ax.annotate("", xy=(11.0, flow_y), xytext=(10.35, flow_y),
                arrowprops=dict(arrowstyle="->", color="#94a3b8", lw=1.0))
    ax.text(11.2, flow_y, "isolated AUROC 诊断反馈", ha="left", fontsize=8, color="#db2777", weight="bold")

    save_all(fig, "fig3_1_method_overview")


def main() -> None:
    draw_mismatch_phase()
    draw_forgetting_bar()
    draw_qualitative_grid()
    draw_method_overview()


if __name__ == "__main__":
    main()
