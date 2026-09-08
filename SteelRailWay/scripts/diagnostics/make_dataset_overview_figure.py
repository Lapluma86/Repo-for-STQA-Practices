#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate Figure 4-0: Rail bimodal multi-view dataset sample grid.

Layout: 3 columns (Cam1 / Cam4 / Cam5) x 2 rows (Normal / Abnormal)
Each cell shows: RGB + Depth side-by-side, with GT mask overlay for abnormal.

Paths:
  RGB:   rail_mvtec_gt_test/rail_mvtec/{cam}/test/{good|broken}/{frame}.jpg
  Depth: rail_mvtec_gt_test/rail_mvtec_depth/{cam}/test/{good|broken}/{frame}.tiff
  GT:    rail_mvtec_gt_test/rail_mvtec/{cam}/ground_truth/broken/{frame}.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "figures"
DATA_RGB = ROOT / "rail_mvtec_gt_test" / "rail_mvtec"
DATA_DEPTH = ROOT / "rail_mvtec_gt_test" / "rail_mvtec_depth"

# ── Selected representative frames ──
FRAMES: list[dict] = [
    # Cam1 Normal
    {"cam": "cam1", "label": "good", "frame": "20251210_185619_Cam1_00010",
     "cam_label": "Cam1", "row": "Normal"},
    # Cam1 Abnormal
    {"cam": "cam1", "label": "broken", "frame": "20251112_191827_Cam1_00015",
     "cam_label": "Cam1", "row": "Abnormal"},
    # Cam4 Normal
    {"cam": "cam4", "label": "good", "frame": "20250417_123456_Cam4_00079",
     "cam_label": "Cam4", "row": "Normal"},
    # Cam4 Abnormal
    {"cam": "cam4", "label": "broken", "frame": "20251210_185619_Cam4_00024",
     "cam_label": "Cam4", "row": "Abnormal"},
    # Cam5 Normal
    {"cam": "cam5", "label": "good", "frame": "20251112_191827_Cam5_00036",
     "cam_label": "Cam5", "row": "Normal"},
    # Cam5 Abnormal
    {"cam": "cam5", "label": "broken", "frame": "20251112_191827_Cam5_00067",
     "cam_label": "Cam5", "row": "Abnormal"},
]


def configure_matplotlib_fonts() -> None:
    """Prefer a CJK-capable font so Chinese titles/captions render correctly."""
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return

    plt.rcParams["axes.unicode_minus"] = False


def load_rgb(cam: str, label: str, frame: str) -> np.ndarray:
    path = DATA_RGB / cam / "test" / label / f"{frame}.jpg"
    if not path.exists():
        raise FileNotFoundError(f"RGB not found: {path}")
    return np.asarray(Image.open(path).convert("RGB"))


def load_depth(cam: str, label: str, frame: str) -> np.ndarray:
    path = DATA_DEPTH / cam / "test" / label / f"{frame}.tiff"
    if not path.exists():
        raise FileNotFoundError(f"Depth not found: {path}")
    depth = np.asarray(Image.open(path))
    # Normalize 16-bit depth to 8-bit for display
    dmin, dmax = np.percentile(depth[depth > 0], [0.5, 99.5]) if (depth > 0).any() else (depth.min(), depth.max())
    depth_clipped = np.clip(depth, dmin, dmax).astype(np.float32)
    depth_norm = ((depth_clipped - dmin) / max(dmax - dmin, 1.0)) * 255.0
    depth_8bit = depth_norm.astype(np.uint8)
    return np.stack([depth_8bit] * 3, axis=-1)  # grayscale → RGB


def load_gt_mask(cam: str, frame: str) -> np.ndarray | None:
    """Load ground truth mask. Returns None if not found."""
    path = DATA_RGB / cam / "ground_truth" / "broken" / f"{frame}.png"
    if not path.exists():
        return None
    mask = np.asarray(Image.open(path).convert("L"))
    return mask


def overlay_mask(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Overlay GT mask contour on RGB image."""
    overlay = rgb.copy()
    mask_bin = (mask > 128).astype(np.uint8)
    from scipy import ndimage
    dilated = ndimage.binary_dilation(mask_bin, iterations=2)
    eroded = ndimage.binary_erosion(mask_bin, iterations=1)
    contour = dilated.astype(np.uint8) - eroded.astype(np.uint8)
    # Red outline where contour is 1
    overlay[contour > 0, 0] = 255
    overlay[contour > 0, 1] = 50
    overlay[contour > 0, 2] = 50
    return overlay


def make_figure(output_path: Path | None = None, dpi: int = 300) -> plt.Figure:
    """Build the 6-panel dataset sample grid."""
    configure_matplotlib_fonts()
    n_cols = 3
    n_rows = 2
    # Each cell has 2 sub-panels: RGB | Depth
    sub_per_cell = 2

    fig = plt.figure(figsize=(16, 11))
    fig.suptitle("图 4-0  钢轨双模态多视角数据集样例",
                 fontsize=16, weight="bold", y=0.985)

    # Create a GridSpec: 2 rows x 6 columns (3 cameras x 2 modalities)
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(n_rows, n_cols * sub_per_cell,
                           figure=fig, wspace=0.06, hspace=0.18,
                           left=0.04, right=0.98, top=0.94, bottom=0.04)

    cam_order = ["cam1", "cam4", "cam5"]
    row_order = ["Normal", "Abnormal"]

    for row_idx, row_label in enumerate(row_order):
        for col_idx, cam in enumerate(cam_order):
            entry = next(f for f in FRAMES
                        if f["cam"] == cam and f["row"] == row_label)
            rgb = load_rgb(entry["cam"], entry["label"], entry["frame"])
            depth = load_depth(entry["cam"], entry["label"], entry["frame"])

            # RGB panel
            ax_rgb = fig.add_subplot(gs[row_idx, col_idx * 2])
            ax_rgb.imshow(rgb)
            ax_rgb.set_axis_off()

            # Depth panel
            ax_depth = fig.add_subplot(gs[row_idx, col_idx * 2 + 1])
            ax_depth.imshow(depth)
            ax_depth.set_axis_off()

            # GT mask overlay for abnormal samples
            if entry["label"] == "broken":
                mask = load_gt_mask(entry["cam"], entry["frame"])
                if mask is not None:
                    rgb_overlay = overlay_mask(rgb, mask)
                    ax_rgb.clear()
                    ax_rgb.imshow(rgb_overlay)
                    ax_rgb.set_axis_off()
                    # Also show mask on depth for reference
                    depth_overlay = overlay_mask(depth, mask)
                    ax_depth.clear()
                    ax_depth.imshow(depth_overlay)
                    ax_depth.set_axis_off()

            # Column/camera title (top row only)
            if row_idx == 0:
                ax_rgb.set_title(f"{entry['cam_label']}\nRGB", fontsize=11,
                                 weight="bold", pad=6)
                ax_depth.set_title(f"{entry['cam_label']}\nDepth", fontsize=11,
                                   weight="bold", pad=6)

            # Row label (left side)
            if col_idx == 0:
                # Add a row label via text
                ax_rgb.text(-0.12, 0.5, row_label,
                           transform=ax_rgb.transAxes,
                           fontsize=12, weight="bold",
                           color="#0f172a" if row_label == "Normal" else "#dc2626",
                           va="center", ha="right", rotation=90)

            # Annotation for abnormal: "GT mask overlay"
            if entry["label"] == "broken" and col_idx == 0:
                ax_rgb.text(0.02, 0.98, "GT mask (red contour)",
                           transform=ax_rgb.transAxes,
                           fontsize=8, color="#dc2626", weight="bold",
                           va="top", ha="left",
                           bbox=dict(facecolor="white", alpha=0.85,
                                    edgecolor="#dc2626", pad=2))

    # ── Global caption ──
    fig.text(0.5, 0.01,
             ("三列分别为 Cam1 / Cam4 / Cam5 视角, 上行正常样本 / 下行异常样本; "
              "每格内左为 RGB 图、右为对应 Depth 图; "
              "异常样本同时叠加 GT 缺陷掩膜 (红色轮廓). "
              "三个视角在成像距离、光照和表面纹理上存在明显差异, "
              "其中 Cam4 为主要适配目标视角."),
             ha="center", fontsize=9, color="#475569", style="italic")

    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate dataset overview figure (Fig 4-0).")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--out_dir", type=str,
                        default=str(FIG_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    svg_path = out_dir / "fig4_0_dataset_samples.svg"
    png_path = out_dir / "fig4_0_dataset_samples.png"

    fig = make_figure(output_path=svg_path, dpi=args.dpi)
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    print(f"  Wrote {svg_path}")
    print(f"  Wrote {png_path}")


if __name__ == "__main__":
    main()
