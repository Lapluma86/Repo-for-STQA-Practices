#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate publication-quality SVG from drawio layout definitions.

This works alongside make_drawio_figures.py: the .drawio files are the
canonical editable source; this script renders clean SVG/PNG for the paper.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "figures"

NS = "http://www.w3.org/2000/svg"

# ═══════════════════════════════════════════════════
# Color palette (same as drawio definitions)
# ═══════════════════════════════════════════════════
FROZEN_FILL = "#ecfdf5"
FROZEN_STROKE = "#059669"
FROZEN_FONT = "#047857"
FROZEN_SUB = "#d1fae5"
TRAIN_FILL = "#eff6ff"
TRAIN_STROKE = "#2563eb"
TRAIN_FONT = "#1d4ed8"
CFCA_FILL = "#e0f2fe"
CFCA_STROKE = "#0284c7"
CFCA_FONT = "#0369a1"
DIAG_FILL = "#fdf2f8"
DIAG_STROKE = "#db2777"
DIAG_FONT = "#9d174d"
OUTPUT_FILL = "#fef3c7"
OUTPUT_STROKE = "#d97706"
OUTPUT_FONT = "#92400e"
INPUT_FILL = "#f8fafc"
INPUT_STROKE = "#475569"
INPUT_FONT = "#0f172a"
PARAM_FILL = "#f5f3ff"
PARAM_STROKE = "#7c3aed"
ARROW_COLOR = "#334155"
GRAY = "#64748b"
DARK = "#0f172a"
WHITE = "#ffffff"
FONT_STACK = "'Microsoft YaHei', 'PingFang SC', 'Noto Sans CJK SC', Arial, Helvetica, sans-serif"


class SVG:
    """Minimal SVG builder."""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.root = ET.Element(
            "svg",
            {
                "xmlns": NS,
                "viewBox": f"0 0 {width} {height}",
                "width": str(width),
                "height": str(height),
            },
        )
        self._defs = ET.SubElement(self.root, "defs")
        self._def_id = 0

    def _uid(self) -> str:
        self._def_id += 1
        return f"d{self._def_id}"

    def add_marker(self, color: str, marker_id: str) -> str:
        mid = marker_id or self._uid()
        m = ET.SubElement(
            self._defs,
            "marker",
            {
                "id": mid,
                "markerWidth": "10",
                "markerHeight": "10",
                "refX": "9",
                "refY": "5",
                "orient": "auto",
                "markerUnits": "strokeWidth",
            },
        )
        ET.SubElement(
            m,
            "path",
            {"d": "M 0 0 L 10 5 L 0 10 z", "fill": color},
        )
        return mid

    def add_arrow_markers(self) -> dict[str, str]:
        markers = {}
        for name, color in [
            ("arrow_gray", ARROW_COLOR),
            ("arrow_blue", TRAIN_STROKE),
            ("arrow_green", FROZEN_STROKE),
            ("arrow_cyan", CFCA_STROKE),
            ("arrow_pink", DIAG_STROKE),
            ("arrow_orange", OUTPUT_STROKE),
        ]:
            markers[name] = self.add_marker(color, name)
        return markers

    def rect(self, x: float, y: float, w: float, h: float,
             fill: str, stroke: str, stroke_width: float = 1.5,
             rx: float = 6, class_: str = "", dashed: bool = False,
             **extra) -> ET.Element:
        attrs = {
            "x": str(x), "y": str(y),
            "width": str(w), "height": str(h),
            "fill": fill, "stroke": stroke,
            "stroke-width": str(stroke_width),
            "rx": str(rx),
        }
        if dashed:
            attrs["stroke-dasharray"] = "8,4"
        if class_:
            attrs["class"] = class_
        attrs.update(extra)
        return ET.SubElement(self.root, "rect", attrs)

    def text(self, x: float, y: float, text: str,
             font_size: float = 12, fill: str = DARK,
             weight: str = "normal", anchor: str = "middle",
             class_: str = "", font_family: str = FONT_STACK,
             **extra) -> ET.Element:
        attrs = {
            "x": str(x), "y": str(y),
            "font-size": str(font_size),
            "fill": fill,
            "font-weight": weight,
            "text-anchor": anchor,
            "font-family": font_family,
        }
        if class_:
            attrs["class"] = class_
        attrs.update(extra)
        el = ET.SubElement(self.root, "text", attrs)
        el.text = text
        return el

    def html_text(self, x: float, y: float, html: str,
                  font_size: float = 12, anchor: str = "middle",
                  **extra) -> ET.Element:
        """Rich text via foreignObject (renders in browsers, not all PDF tools)."""
        fo = ET.SubElement(
            self.root,
            "foreignObject",
            {
                "x": str(x - 200), "y": str(y - font_size),
                "width": "400", "height": str(font_size * 3 + 20),
            },
        )
        div = ET.SubElement(fo, "div", {"xmlns": NS if False else ""})
        div.text = html
        return fo

    def line(self, x1: float, y1: float, x2: float, y2: float,
             stroke: str = ARROW_COLOR, stroke_width: float = 1.5,
             marker_end: str = "", **extra) -> ET.Element:
        attrs = {
            "x1": str(x1), "y1": str(y1),
            "x2": str(x2), "y2": str(y2),
            "stroke": stroke,
            "stroke-width": str(stroke_width),
        }
        if marker_end:
            attrs["marker-end"] = f"url(#{marker_end})"
        attrs.update(extra)
        return ET.SubElement(self.root, "line", attrs)

    def arrow(self, x1: float, y1: float, x2: float, y2: float,
              color: str = ARROW_COLOR, marker: str = "arrow_gray",
              width: float = 1.8):
        return self.line(x1, y1, x2, y2, stroke=color,
                         stroke_width=width, marker_end=marker)

    def arrow_h(self, x1: float, x2: float, y: float,
                color: str = ARROW_COLOR, marker: str = "arrow_gray"):
        return self.arrow(x1, y, x2, y, color=color, marker=marker)

    def multiline_text(self, x: float, y: float, lines: list[tuple[str, float, str]],
                       default_size: float = 12, default_fill: str = DARK,
                       default_weight: str = "normal", anchor: str = "middle",
                       line_height: float = 1.35):
        """Draw multiple text lines. Each line: (text, font_size, fill_or_weight)."""
        for i, line_data in enumerate(lines):
            text, size, fill_or_weight = line_data[0], line_data[1] if len(line_data) > 1 else default_size, line_data[2] if len(line_data) > 2 else default_fill
            actual_weight = fill_or_weight if fill_or_weight == "bold" else default_weight
            actual_fill = fill_or_weight if fill_or_weight.startswith("#") else default_fill
            self.text(
                x, y + i * default_size * line_height,
                text, font_size=size, fill=actual_fill,
                weight=actual_weight, anchor=anchor,
            )

    def save(self, path: Path) -> None:
        tree = ET.ElementTree(self.root)
        path.parent.mkdir(parents=True, exist_ok=True)
        tree.write(str(path), encoding="UTF-8", xml_declaration=True)
        print(f"  Wrote {path}")


# ═══════════════════════════════════════════════════
# Figure 3-1: Overall method pipeline
# ═══════════════════════════════════════════════════

def draw_fig3_1_svg() -> SVG:
    W, H = 1600, 820
    svg = SVG(W, H)
    mk = svg.add_arrow_markers()

    # Background
    svg.rect(0, 0, W, H, WHITE, "none", 0, rx=0)

    # ── Title ──
    svg.text(W / 2, 32, "图 3-1  整体方法框架图",
             font_size=20, weight="bold", fill=DARK)

    # ── Legend (top) ──
    lx, ly = 30, 50
    for i, (label, fill, stroke, font_c) in enumerate([
        ("冻结模块", FROZEN_FILL, FROZEN_STROKE, FROZEN_FONT),
        ("可训练模块", TRAIN_FILL, TRAIN_STROKE, TRAIN_FONT),
        ("选择性微调 (Selective FT)", CFCA_FILL, CFCA_STROKE, CFCA_FONT),
        ("诊断模块", DIAG_FILL, DIAG_STROKE, DIAG_FONT),
    ]):
        ox = lx + i * 170
        svg.rect(ox, ly, 22, 16, fill, stroke, 1.5, rx=3)
        svg.text(ox + 28, ly + 12, label, font_size=11, fill=font_c, anchor="start")

    # ── Section labels ──
    sections = [
        (90, "① 双模态输入"),
        (420, "② 输入端修正"),
        (680, "③ 冻结主干推理 + 协同修复"),
        (1160, "④ 异常检测输出"),
    ]
    for sx, label in sections:
        svg.text(sx, 78, label, font_size=10, fill=GRAY)

    # ── ① Inputs ──
    ix, iy = 35, 95
    iw, ih = 185, 80

    # RGB input
    svg.rect(ix, iy, iw, ih, INPUT_FILL, INPUT_STROKE, 1.8)
    svg.text(ix + iw / 2, iy + 28, "Intensity", font_size=13, weight="bold")
    svg.text(ix + iw / 2, iy + 54, "(RGB Image)", font_size=11, fill=GRAY)

    # Depth input
    dy = iy + 280
    svg.rect(ix, dy, iw, ih, INPUT_FILL, INPUT_STROKE, 1.8)
    svg.text(ix + iw / 2, dy + 28, "Depth", font_size=13, weight="bold")
    svg.text(ix + iw / 2, dy + 54, "(16-bit TIFF)", font_size=11, fill=GRAY)

    # ── ② DepthAffinePEFT ──
    px = 320
    svg.rect(px, dy, 215, ih, TRAIN_FILL, TRAIN_STROKE, 2.5)
    svg.text(px + 107, dy + 24, "DepthAffinePEFT", font_size=12, weight="bold", fill=TRAIN_FONT)
    svg.text(px + 107, dy + 50, "xₗ' = γ · xₗ + β", font_size=11, fill=DARK)
    svg.text(px + 107, dy + 70, "(2 trainable scalars)", font_size=9, fill=TRAIN_STROKE)

    # Arrow: Depth → PEFT
    svg.arrow_h(ix + iw, px, dy + ih / 2, color=ARROW_COLOR, marker="arrow_gray")

    # ── ③ TRD Backbone (large enclosure) ──
    bb_x, bb_y, bb_w, bb_h = 610, 95, 520, 400
    svg.rect(bb_x, bb_y, bb_w, bb_h, FROZEN_FILL, FROZEN_STROKE, 3, rx=10)
    svg.text(bb_x + bb_w / 2, bb_y + 26, "Frozen TRD Backbone",
             font_size=14, weight="bold", fill=FROZEN_FONT)
    svg.text(bb_x + bb_w / 2, bb_y + 46,
             "Teacher-Student Distillation + Cross-modal Fusion",
             font_size=9, fill=FROZEN_FONT)

    # Sub-modules inside backbone: 2x2 grid
    sw, sh = 210, 80
    sx0 = bb_x + 30
    sy0 = bb_y + 65
    gap_x, gap_y = 30, 20

    # Row 1: Teacher Encoder + Student Decoder (frozen)
    svg.rect(sx0, sy0, sw, sh, FROZEN_SUB, FROZEN_STROKE, 1.5)
    svg.text(sx0 + sw / 2, sy0 + 32, "Teacher Encoder", font_size=11, weight="bold", fill=FROZEN_FONT)
    svg.text(sx0 + sw / 2, sy0 + 55, "WRN50_2 (frozen)", font_size=9, fill=FROZEN_FONT)

    svg.rect(sx0 + sw + gap_x, sy0, sw, sh, FROZEN_SUB, FROZEN_STROKE, 1.5)
    svg.text(sx0 + sw + gap_x + sw / 2, sy0 + 32, "Student Decoder", font_size=11, weight="bold", fill=FROZEN_FONT)
    svg.text(sx0 + sw + gap_x + sw / 2, sy0 + 55, "(frozen)", font_size=9, fill=FROZEN_FONT)

    # Row 2: CF + CA (selective FT)
    sy2 = sy0 + sh + gap_y
    svg.rect(sx0, sy2, sw, sh, CFCA_FILL, CFCA_STROKE, 2.5)
    svg.text(sx0 + sw / 2, sy2 + 28, "CF Module", font_size=11, weight="bold", fill=CFCA_FONT)
    svg.text(sx0 + sw / 2, sy2 + 50, "Crossmodal Filter", font_size=9, fill=CFCA_FONT)
    svg.text(sx0 + sw / 2, sy2 + 66, "(Selective FT)", font_size=8, fill=CFCA_STROKE)

    svg.rect(sx0 + sw + gap_x, sy2, sw, sh, CFCA_FILL, CFCA_STROKE, 2.5)
    svg.text(sx0 + sw + gap_x + sw / 2, sy2 + 28, "CA Module", font_size=11, weight="bold", fill=CFCA_FONT)
    svg.text(sx0 + sw + gap_x + sw / 2, sy2 + 50, "Crossmodal Amplifier", font_size=9, fill=CFCA_FONT)
    svg.text(sx0 + sw + gap_x + sw / 2, sy2 + 66, "(Selective FT)", font_size=8, fill=CFCA_STROKE)

    # Annotation arrow for selective FT
    svg.text(sx0 + sw / 2, sy2 - 13, "CF/CA Selective FT", font_size=8, fill=CFCA_FONT, weight="bold")

    # Note below backbone
    svg.text(bb_x + bb_w / 2, bb_y + bb_h + 14,
             "除 CF/CA 选择性微调参数外全部冻结  |  总计可训练 ≈ 110 参数",
             font_size=9, fill=FROZEN_FONT)

    # Arrow: PEFT → Backbone
    svg.arrow_h(px + 215, bb_x, dy + ih / 2, color=TRAIN_STROKE, marker="arrow_blue")

    # Arrow: RGB → Backbone (via above)
    mid_y = iy + ih / 2
    svg.arrow(ix + iw, mid_y, ix + iw + 30, mid_y, color=ARROW_COLOR, marker="arrow_gray")
    svg.arrow(ix + iw + 30, mid_y, ix + iw + 30, bb_y + bb_h / 2 - 20, color=ARROW_COLOR)
    svg.arrow(ix + iw + 30, bb_y + bb_h / 2 - 20, bb_x, bb_y + bb_h / 2 - 20, color=ARROW_COLOR, marker="arrow_gray")

    # ── ④ Fusion Output ──
    out_x = 1220
    out_w, out_h = 280, 110
    out_y = iy + 170
    svg.rect(out_x, out_y, out_w, out_h, OUTPUT_FILL, OUTPUT_STROKE, 2)
    svg.text(out_x + out_w / 2, out_y + 28, "Fusion Anomaly Map", font_size=13, weight="bold", fill=OUTPUT_FONT)
    svg.text(out_x + out_w / 2, out_y + 55, "M = Mᵢⁿᵒʳᵐ + Mₗⁿᵒʳᵐ",
             font_size=11, fill=DARK)
    svg.text(out_x + out_w / 2, out_y + 80, "normalized sum fusion", font_size=9, fill=OUTPUT_STROKE)

    # Arrow: backbone → output
    svg.arrow_h(bb_x + bb_w, out_x, bb_y + bb_h / 2, color=ARROW_COLOR, marker="arrow_gray")

    # ── Isolated AUROC Diagnostic (top bar) ──
    diag_y = bb_y + bb_h + 15
    svg.rect(bb_x, diag_y, bb_w, 38, DIAG_FILL, DIAG_STROKE, 2, dashed=True)
    svg.text(bb_x + bb_w / 2, diag_y + 24,
             "isolated AUROC 诊断:  Δ = AUROC_cross − AUROC_isolated  →  量化跨模态路径贡献度",
             font_size=11, weight="bold", fill=DIAG_FONT)

    # ── Parameter budget (right side) ──
    param_y = diag_y + 50
    svg.rect(out_x, param_y, out_w, 160, PARAM_FILL, PARAM_STROKE, 2)
    param_lines = [
        ("训练参数统计", 12, PARAM_STROKE),
        ("", 6, DARK),
        ("DepthAffinePEFT: γ, β  (2 params)", 9, DARK),
        ("CF/CA Selective FT: ≈ 108 params", 9, DARK),
        ("主干参数: 0 (全部冻结)", 9, DARK),
        ("", 6, DARK),
        ("总计可训练: ≈ 110 个", 10, PARAM_STROKE),
        ("占主干比例 ≈ 10⁻⁶", 8, GRAY),
    ]
    for i, (line, size, color) in enumerate(param_lines):
        w = "bold" if i == 0 or i == 6 else "normal"
        svg.text(out_x + out_w / 2, param_y + 22 + i * 18, line,
                 font_size=size, fill=color, weight=w)

    # ── CAD patch management (bottom) ──
    cad_y = param_y + 175
    cad_w = W - 60
    svg.rect(30, cad_y, cad_w, 52, "#f8fafc", "#94a3b8", 1.5, dashed=True)
    svg.text(W / 2, cad_y + 22,
             "CAD 补丁式持续适配管理",
             font_size=12, weight="bold", fill=GRAY)
    svg.text(W / 2, cad_y + 42,
             "每个任务 Tₖ 生成独立补丁 Pₖ = {φᵏ_affine, φᵏ_cf/ca, mₖ}  →  补丁仓库, 可独立加载 / 卸载 / 回滚",
             font_size=10, fill=GRAY)

    # ── Flow summary (bottom) ──
    flow_y = cad_y + 68
    flow = [
        ("输入端修正 →", TRAIN_FONT),
        ("冻结主干推理 →", FROZEN_FONT),
        ("协同选择性微调 →", CFCA_FONT),
        ("融合异常图 →", OUTPUT_FONT),
        ("isolated AUROC 诊断反馈", DIAG_FONT),
    ]
    fx = 130
    for label, color in flow:
        svg.text(fx, flow_y, label, font_size=10, fill=color, weight="bold")
        fx += len(label) * 14 + 15

    return svg


# ═══════════════════════════════════════════════════
# Figure 3-2: Detail mechanism diagram
# ═══════════════════════════════════════════════════

def draw_fig3_2_svg() -> SVG:
    W, H = 1520, 820
    svg = SVG(W, H)
    mk = svg.add_arrow_markers()

    svg.rect(0, 0, W, H, WHITE, "none", 0, rx=0)

    # ── Title ──
    svg.text(W / 2, 30, "图 3-2  PEFT + CF/CA 协同选择性微调细节机制图",
             font_size=20, weight="bold", fill=DARK)

    # ── Branch labels ──
    branch_x = 15
    svg.text(branch_x + 90, 82, "Intensity Branch",
             font_size=14, weight="bold", fill=FROZEN_FONT)
    svg.text(branch_x + 90, 448, "Depth Branch (with PEFT)",
             font_size=14, weight="bold", fill=TRAIN_FONT)

    # ── Intensity Branch ──
    iby = 100
    box_w, box_h = 185, 70
    small_w, small_h = 150, 60

    # I Input
    svg.rect(branch_x, iby, box_w, box_h, INPUT_FILL, INPUT_STROKE, 1.8)
    svg.text(branch_x + box_w / 2, iby + 26, "Intensity", font_size=12, weight="bold")
    svg.text(branch_x + box_w / 2, iby + 50, "RGB Image  xᵢ", font_size=10, fill=GRAY)

    # I Encoder (frozen)
    enc_x = 260
    svg.rect(enc_x, iby, box_w + 30, box_h, FROZEN_FILL, FROZEN_STROKE, 1.8)
    svg.text(enc_x + box_w / 2 + 15, iby + 22, "Teacher Encoder", font_size=11, weight="bold", fill=FROZEN_FONT)
    svg.text(enc_x + box_w / 2 + 15, iby + 42, "WRN50_2 (frozen)", font_size=9, fill=FROZEN_FONT)
    svg.text(enc_x + box_w / 2 + 15, iby + 58, "multi-scale features fᵢˡ", font_size=9, fill=GRAY)

    svg.arrow_h(branch_x + box_w, enc_x, iby + box_h / 2)

    # I Bottleneck (frozen)
    bn_x = 510
    svg.rect(bn_x, iby + 100, small_w, small_h, FROZEN_FILL, FROZEN_STROKE, 1.8)
    svg.text(bn_x + small_w / 2, iby + 100 + 22, "I-Bottleneck", font_size=10, weight="bold", fill=FROZEN_FONT)
    svg.text(bn_x + small_w / 2, iby + 100 + 42, "compressed rep", font_size=9, fill=GRAY)

    # I Enc → Bottleneck
    svg.arrow(enc_x + box_w + 30, iby + box_h / 2, enc_x + box_w + 50, iby + box_h / 2, color=ARROW_COLOR)
    svg.arrow(enc_x + box_w + 50, iby + box_h / 2, enc_x + box_w + 50, iby + 100 + small_h / 2, color=ARROW_COLOR)
    svg.arrow(enc_x + box_w + 50, iby + 100 + small_h / 2, bn_x, iby + 100 + small_h / 2, color=ARROW_COLOR, marker="arrow_gray")

    # I Student Decoder (frozen)
    dec_x = 720
    svg.rect(dec_x, iby + 100, box_w, box_h, FROZEN_FILL, FROZEN_STROKE, 1.8)
    svg.text(dec_x + box_w / 2, iby + 100 + 22, "Student Decoder", font_size=11, weight="bold", fill=FROZEN_FONT)
    svg.text(dec_x + box_w / 2, iby + 100 + 42, "(frozen)", font_size=9, fill=FROZEN_FONT)
    svg.text(dec_x + box_w / 2, iby + 100 + 58, "recover teacher features", font_size=9, fill=GRAY)

    svg.arrow_h(bn_x + small_w, dec_x, iby + 100 + small_h / 2)

    # I Anomaly Map
    amap_x = 980
    amap_w, amap_h = 195, 80
    svg.rect(amap_x, iby + 90, amap_w, amap_h, OUTPUT_FILL, OUTPUT_STROKE, 1.8)
    svg.text(amap_x + amap_w / 2, iby + 90 + 24, "I-Anomaly Map", font_size=11, weight="bold", fill=OUTPUT_FONT)
    svg.text(amap_x + amap_w / 2, iby + 90 + 46, "Mᵢ = teacher-student diff", font_size=10, fill=DARK)
    svg.text(amap_x + amap_w / 2, iby + 90 + 66, "→ cross-conditioned score", font_size=9, fill=GRAY)

    svg.arrow_h(dec_x + box_w, amap_x, iby + 100 + box_h / 2)

    # ── Depth Branch ──
    dby = 460
    # D Input
    svg.rect(branch_x, dby, box_w, box_h, INPUT_FILL, INPUT_STROKE, 1.8)
    svg.text(branch_x + box_w / 2, dby + 26, "Depth", font_size=12, weight="bold")
    svg.text(branch_x + box_w / 2, dby + 50, "16-bit TIFF  xₗ", font_size=10, fill=GRAY)

    # DepthAffinePEFT (trainable)
    peft_x = branch_x
    peft_y = dby + 85
    peft_h = 75
    svg.rect(peft_x, peft_y, box_w, peft_h, TRAIN_FILL, TRAIN_STROKE, 3)
    svg.text(peft_x + box_w / 2, peft_y + 22, "DepthAffinePEFT", font_size=11, weight="bold", fill=TRAIN_FONT)
    svg.text(peft_x + box_w / 2, peft_y + 44, "xₗ' = γ · xₗ + β", font_size=11, fill=DARK)
    svg.text(peft_x + box_w / 2, peft_y + 64, "(2 trainable scalars)", font_size=9, fill=TRAIN_STROKE)

    svg.arrow(branch_x + box_w / 2, dby + box_h, branch_x + box_w / 2, peft_y, color=TRAIN_STROKE, marker="arrow_blue")

    # D Encoder (frozen)
    svg.rect(enc_x, peft_y, box_w + 30, box_h, FROZEN_FILL, FROZEN_STROKE, 1.8)
    svg.text(enc_x + box_w / 2 + 15, peft_y + 22, "Teacher Encoder", font_size=11, weight="bold", fill=FROZEN_FONT)
    svg.text(enc_x + box_w / 2 + 15, peft_y + 42, "WRN50_2 (frozen)", font_size=9, fill=FROZEN_FONT)
    svg.text(enc_x + box_w / 2 + 15, peft_y + 58, "multi-scale features fₗˡ", font_size=9, fill=GRAY)

    svg.arrow_h(peft_x + box_w, enc_x, peft_y + box_h / 2, color=TRAIN_STROKE, marker="arrow_blue")

    # D Bottleneck (frozen)
    d_bn_y = peft_y + box_h + 20
    svg.rect(bn_x, d_bn_y, small_w, small_h, FROZEN_FILL, FROZEN_STROKE, 1.8)
    svg.text(bn_x + small_w / 2, d_bn_y + 22, "D-Bottleneck", font_size=10, weight="bold", fill=FROZEN_FONT)
    svg.text(bn_x + small_w / 2, d_bn_y + 42, "compressed rep", font_size=9, fill=GRAY)

    svg.arrow(enc_x + box_w + 30, peft_y + box_h / 2, enc_x + box_w + 50, peft_y + box_h / 2, color=ARROW_COLOR)
    svg.arrow(enc_x + box_w + 50, peft_y + box_h / 2, enc_x + box_w + 50, d_bn_y + small_h / 2, color=ARROW_COLOR)
    svg.arrow(enc_x + box_w + 50, d_bn_y + small_h / 2, bn_x, d_bn_y + small_h / 2, color=ARROW_COLOR, marker="arrow_gray")

    # D Student Decoder (frozen)
    svg.rect(dec_x, d_bn_y, box_w, box_h, FROZEN_FILL, FROZEN_STROKE, 1.8)
    svg.text(dec_x + box_w / 2, d_bn_y + 22, "Student Decoder", font_size=11, weight="bold", fill=FROZEN_FONT)
    svg.text(dec_x + box_w / 2, d_bn_y + 42, "(frozen)", font_size=9, fill=FROZEN_FONT)
    svg.text(dec_x + box_w / 2, d_bn_y + 58, "recover teacher features", font_size=9, fill=GRAY)

    svg.arrow_h(bn_x + small_w, dec_x, d_bn_y + small_h / 2)

    # D Anomaly Map
    svg.rect(amap_x, d_bn_y, amap_w, amap_h, OUTPUT_FILL, OUTPUT_STROKE, 1.8)
    svg.text(amap_x + amap_w / 2, d_bn_y + 24, "D-Anomaly Map", font_size=11, weight="bold", fill=OUTPUT_FONT)
    svg.text(amap_x + amap_w / 2, d_bn_y + 46, "Mₗ = teacher-student diff", font_size=10, fill=DARK)
    svg.text(amap_x + amap_w / 2, d_bn_y + 66, "→ cross-conditioned score", font_size=9, fill=GRAY)

    svg.arrow_h(dec_x + box_w, amap_x, d_bn_y + box_h / 2)

    # ── CF Module (bottleneck cross-modal) ──
    cf_x = bn_x + small_w + 35
    cf_y = (iby + 100 + d_bn_y + small_h) / 2 - 55
    cf_w, cf_h = 140, 95
    svg.rect(cf_x, cf_y, cf_w, cf_h, CFCA_FILL, CFCA_STROKE, 2.8)
    svg.text(cf_x + cf_w / 2, cf_y + 22, "CF Module", font_size=11, weight="bold", fill=CFCA_FONT)
    svg.text(cf_x + cf_w / 2, cf_y + 42, "Crossmodal Filter", font_size=9, fill=CFCA_FONT)
    svg.text(cf_x + cf_w / 2, cf_y + 58, "at bottleneck", font_size=9, fill=CFCA_FONT)
    svg.text(cf_x + cf_w / 2, cf_y + 74, "Selective FT ≈ 54p", font_size=8, fill=CFCA_STROKE)

    # CF bidirectional arrows
    svg.arrow(bn_x + small_w, iby + 100 + small_h / 2, cf_x, cf_y + cf_h / 3,
              color=CFCA_STROKE, marker="arrow_cyan", width=1.5)
    svg.arrow(bn_x + small_w, d_bn_y + small_h / 2, cf_x, cf_y + 2 * cf_h / 3,
              color=CFCA_STROKE, marker="arrow_cyan", width=1.5)
    svg.arrow(cf_x + cf_w, cf_y + cf_h / 3, bn_x + small_w + 8, iby + 100 + small_h / 2,
              color=CFCA_STROKE, marker="arrow_cyan", width=1.5)
    svg.arrow(cf_x + cf_w, cf_y + 2 * cf_h / 3, bn_x + small_w + 8, d_bn_y + small_h / 2,
              color=CFCA_STROKE, marker="arrow_cyan", width=1.5)

    svg.text(bn_x + small_w + 15, (iby + 100 + small_h / 2 + cf_y + cf_h / 3) / 2,
             "I↔D", font_size=8, fill=CFCA_FONT)

    # ── CA Module (decoder skip cross-modal) ──
    ca_x = dec_x + box_w + 30
    ca_y = cf_y
    ca_w, ca_h = 150, 95
    svg.rect(ca_x, ca_y, ca_w, ca_h, CFCA_FILL, CFCA_STROKE, 2.8)
    svg.text(ca_x + ca_w / 2, ca_y + 22, "CA Module", font_size=11, weight="bold", fill=CFCA_FONT)
    svg.text(ca_x + ca_w / 2, ca_y + 42, "Crossmodal Amplifier", font_size=9, fill=CFCA_FONT)
    svg.text(ca_x + ca_w / 2, ca_y + 58, "at decoder skip fusion", font_size=9, fill=CFCA_FONT)
    svg.text(ca_x + ca_w / 2, ca_y + 74, "Selective FT ≈ 54p", font_size=8, fill=CFCA_STROKE)

    # CA bidirectional arrows
    svg.arrow(dec_x + box_w, iby + 100 + box_h / 2, ca_x, ca_y + ca_h / 3,
              color=CFCA_STROKE, marker="arrow_cyan", width=1.5)
    svg.arrow(dec_x + box_w, d_bn_y + box_h / 2, ca_x, ca_y + 2 * ca_h / 3,
              color=CFCA_STROKE, marker="arrow_cyan", width=1.5)
    svg.arrow(ca_x + ca_w, ca_y + ca_h / 3, dec_x + box_w + 8, iby + 100 + box_h / 2,
              color=CFCA_STROKE, marker="arrow_cyan", width=1.5)
    svg.arrow(ca_x + ca_w, ca_y + 2 * ca_h / 3, dec_x + box_w + 8, d_bn_y + box_h / 2,
              color=CFCA_STROKE, marker="arrow_cyan", width=1.5)

    # ── Fusion Output ──
    fusion_x = amap_x + amap_w + 50
    fusion_y = (iby + 90 + d_bn_y + amap_h) / 2 - 55
    fusion_w, fusion_h = 230, 110
    svg.rect(fusion_x, fusion_y, fusion_w, fusion_h, OUTPUT_FILL, OUTPUT_STROKE, 2.5)
    svg.text(fusion_x + fusion_w / 2, fusion_y + 22, "Fusion Anomaly Map", font_size=12, weight="bold", fill=OUTPUT_FONT)
    svg.text(fusion_x + fusion_w / 2, fusion_y + 48, "M = Mᵢⁿᵒʳᵐ + Mₗⁿᵒʳᵐ", font_size=11, fill=DARK)
    svg.text(fusion_x + fusion_w / 2, fusion_y + 72, "normalized sum fusion", font_size=10, fill=OUTPUT_STROKE)
    svg.text(fusion_x + fusion_w / 2, fusion_y + 94, "→ image-level anomaly score", font_size=9, fill=GRAY)

    # Arrows from anomaly maps to fusion
    svg.arrow(amap_x + amap_w, iby + 90 + amap_h / 2, fusion_x, fusion_y + fusion_h / 3,
              color=ARROW_COLOR, marker="arrow_gray")
    svg.arrow(amap_x + amap_w, d_bn_y + amap_h / 2, fusion_x, fusion_y + 2 * fusion_h / 3,
              color=ARROW_COLOR, marker="arrow_gray")

    # ── Phase labels ──
    # "Bottleneck" and "Decoder Skip" tags near CF/CA
    svg.text(bn_x + small_w / 2, iby + 100 - 14, "← Bottleneck stage",
             font_size=9, fill=CFCA_STROKE, weight="bold")
    svg.text(dec_x + box_w / 2, iby + 100 - 14, "← Decoder skip stage",
             font_size=9, fill=CFCA_STROKE, weight="bold")

    # ── Training Loss annotation ──
    loss_y = 710
    svg.rect(15, loss_y, 440, 95, "#fefce8", "#ca8a04", 2)
    svg.text(20, loss_y + 24, "Training Loss", font_size=12, weight="bold", fill="#854d0e", anchor="start")
    svg.text(20, loss_y + 46, "L = L_FDM + λ_cf · L_CF + λ_ca · L_CA", font_size=11, fill=DARK, anchor="start")
    svg.text(20, loss_y + 66, "L_FDM: feature distribution matching (μ, σ)   |   L_CF: CF consistency   |   L_CA: CA consistency",
             font_size=9, fill=GRAY, anchor="start")
    svg.text(20, loss_y + 82, "Only φ_affine (γ, β) and φ_cf/ca updated; all other parameters frozen",
             font_size=9, fill=GRAY, anchor="start")

    # ── Inference annotation ──
    inf_x = 490
    svg.rect(inf_x, loss_y, 460, 95, "#f0fdf4", "#16a34a", 2)
    svg.text(inf_x + 10, loss_y + 24, "Inference (no gradient)", font_size=12, weight="bold", fill="#166534", anchor="start")
    steps = [
        "1. DepthAffinePEFT: xₗ' = γ · xₗ + β",
        "2. Extract teacher features fᵢ, fₗ (multi-scale)",
        "3. CF at bottleneck, CA at decoder skip: cross-modal modulation",
        "4. Student recovery → Mᵢ, Mₗ (anomaly maps)",
        "5. M = Mᵢⁿᵒʳᵐ + Mₗⁿᵒʳᵐ → image score",
    ]
    for i, step in enumerate(steps):
        svg.text(inf_x + 10, loss_y + 46 + i * 14, step, font_size=9, fill=DARK, anchor="start")

    # ── Legend (bottom right) ──
    leg_x = 990
    svg.rect(leg_x, loss_y - 5, 250, 120, "#f8fafc", "#cbd5e1", 1)
    svg.text(leg_x + 10, loss_y + 20, "Legend", font_size=11, weight="bold", fill=GRAY, anchor="start")
    for i, (color, label) in enumerate([
        (FROZEN_STROKE, "Frozen (no gradient)"),
        (TRAIN_STROKE, "Trainable (PEFT params)"),
        (CFCA_STROKE, "Selective Fine-Tuning"),
        (ARROW_COLOR, "Data flow"),
        ("#ca8a04", "Training signal"),
    ]):
        svg.rect(leg_x + 10, loss_y + 28 + i * 17, 14, 12, color, color, 1, rx=2)
        svg.text(leg_x + 32, loss_y + 38 + i * 17, label, font_size=9, fill=DARK, anchor="start")

    return svg


# ═══════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════

def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    for name, builder in [
        ("fig3_1_method_pipeline", draw_fig3_1_svg),
        ("fig3_2_peft_cfca_detail", draw_fig3_2_svg),
    ]:
        svg = builder()
        svg.save(FIG_DIR / f"{name}.svg")

    # Convert SVG to PNG using cairosvg
    try:
        import cairosvg
        for name in ["fig3_1_method_pipeline", "fig3_2_peft_cfca_detail"]:
            svg_path = FIG_DIR / f"{name}.svg"
            png_path = FIG_DIR / f"{name}.png"
            try:
                cairosvg.svg2png(url=str(svg_path), write_to=str(png_path), dpi=300)
                print(f"  Wrote {png_path}")
            except Exception as e:
                print(f"  WARNING: cairosvg failed for {name}: {e}")
                print(f"  Falling back to PIL rasterization...")
                # Fallback: render via cairo from bytes
                try:
                    import cairocffi as cairo
                    svg_data = svg_path.read_bytes()
                    cairosvg.svg2png(bytestring=svg_data, write_to=str(png_path), dpi=150)
                    print(f"  Wrote {png_path} (cairo fallback)")
                except Exception as e2:
                    print(f"  PNG export failed for {name}: {e2}")
    except ImportError:
        print("  NOTE: cairosvg not available, PNG export skipped.")

    # Also regenerate fig3_1 via existing matplotlib approach as backup
    try:
        import matplotlib
        matplotlib.use("Agg")
        from make_thesis_figures import draw_method_overview
        draw_method_overview()
        print("  Regenerated fig3_1 via matplotlib (backup)")
    except Exception as e:
        print(f"  Matplotlib fallback skipped: {e}")


if __name__ == "__main__":
    main()
