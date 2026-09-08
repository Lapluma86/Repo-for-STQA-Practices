#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate paper-quality .drawio files for thesis Chapter 3 figures.

Produces:
  figures/fig3_1_method_pipeline.drawio  -- overall method framework
  figures/fig3_2_peft_cfca_detail.drawio -- detail mechanism diagram

The .drawio output is MXGraph XML that can be edited in draw.io / VS Code
drawio extension and exported to SVG/PNG via: drawio-export -f svg <file>
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "figures"


# ═══════════════════════════════════════════════════════════════════
# Color palette
# ═══════════════════════════════════════════════════════════════════
FROZEN_FILL = "#ecfdf5"
FROZEN_STROKE = "#059669"
FROZEN_FONT = "#047857"
TRAIN_FILL = "#eff6ff"
TRAIN_STROKE = "#2563eb"
TRAIN_FONT = "#1d4ed8"
INPUT_FILL = "#f8fafc"
INPUT_STROKE = "#475569"
OUTPUT_FILL = "#fef3c7"
OUTPUT_STROKE = "#d97706"
OUTPUT_FONT = "#92400e"
DIAG_FILL = "#fdf2f8"
DIAG_STROKE = "#db2777"
DIAG_FONT = "#9d174d"
CFCA_FILL = "#e0f2fe"
CFCA_STROKE = "#0284c7"
CFCA_FONT = "#0369a1"
PARAM_FILL = "#f5f3ff"
PARAM_STROKE = "#7c3aed"
SUB_FILL = "#d1fae5"
ARROW = "#334155"
GRAY_FONT = "#475569"
WHITE = "#ffffff"

# ═══════════════════════════════════════════════════════════════════
# XML helpers
# ═══════════════════════════════════════════════════════════════════


def _make_mxfile(diagrams: list[ET.Element]) -> ET.Element:
    mxfile = ET.Element("mxfile", {
        "host": "app.diagrams.net",
        "modified": "2026-05-06T00:00:00.000Z",
        "agent": "Python script",
        "version": "24.0.0",
        "type": "device",
    })
    mxfile.extend(diagrams)
    return mxfile


def _make_diagram(name: str, page_id: str = "page-1") -> ET.Element:
    diag = ET.Element("diagram", {"name": name, "id": page_id})
    model = ET.SubElement(diag, "mxGraphModel", {
        "dx": "1422", "dy": "794", "grid": "1", "gridSize": "10",
        "guides": "1", "tooltips": "1", "connect": "1", "arrows": "1",
        "fold": "1", "page": "1", "pageScale": "1",
        "pageWidth": "1600", "pageHeight": "900",
        "math": "0", "shadow": "0",
    })
    root_elem = ET.SubElement(model, "root")
    ET.SubElement(root_elem, "mxCell", {"id": "0"})
    ET.SubElement(root_elem, "mxCell", {"id": "1", "parent": "0"})
    return diag


def _cell(parent_id: str, cell_id: str, value: str, style: str,
          x: float, y: float, w: float, h: float,
          vertex: bool = True, **extra) -> ET.Element:
    cell = ET.Element("mxCell", {
        "id": cell_id, "parent": parent_id,
        "value": value, "style": style,
        "vertex": "1" if vertex else "0",
    })
    geom = ET.SubElement(cell, "mxGeometry", {
        "x": str(x), "y": str(y),
        "width": str(w), "height": str(h),
        "as": "geometry",
    })
    for k, v in extra.items():
        cell.set(k, str(v))
    return cell


def _edge(parent_id: str, cell_id: str, source: str, target: str,
          style: str, value: str = "", **extra) -> ET.Element:
    cell = ET.Element("mxCell", {
        "id": cell_id, "parent": parent_id,
        "source": source, "target": target,
        "value": value, "style": style,
        "edge": "1",
    })
    geom = ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
    for k, v in extra.items():
        cell.set(k, str(v))
    return cell


def _box_style(fill: str, stroke: str, font: str = "#0f172a",
               rounded: int = 1, stroke_width: int = 2,
               align: str = "center", font_size: int = 12,
               vertical_align: str = "middle",
               white_space: str = "wrap",
               dashed: int = 0) -> str:
    parts = [
        f"rounded={rounded}",
        "whiteSpace=wrap",
        f"html=1",
        f"fillColor={fill}",
        f"strokeColor={stroke}",
        f"strokeWidth={stroke_width}",
        f"fontColor={font}",
        f"fontSize={font_size}",
        f"align={align}",
        f"verticalAlign={vertical_align}",
    ]
    if dashed:
        parts.append(f"dashed={dashed}")
    return ";".join(parts)


def _arrow_style(color: str = ARROW, width: int = 2,
                 end_arrow: str = "classic",
                 curved: int = 0) -> str:
    return (f"edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;"
            f"jettySize=auto;html=1;strokeColor={color};"
            f"strokeWidth={width};endArrow={end_arrow};"
            f"endFill=1;exitX=1;exitY=0.5;exitDx=0;exitDy=0;"
            f"entryX=0;entryY=0.5;entryDx=0;entryDy=0;"
            f"curved={curved}")


# ═══════════════════════════════════════════════════════════════════
# Figure 3-1: Overall method framework
# ═══════════════════════════════════════════════════════════════════

def build_fig3_1_elements(root_id: str) -> list[ET.Element]:
    elements: list[ET.Element] = []
    pid = root_id
    cid = 2  # start cell id

    # ── Title ──
    elements.append(_cell(pid, str(cid),
        "图 3-1  整体方法框架图",
        _box_style("none", "none", "#0f172a", font_size=18, rounded=0),
        400, 10, 800, 40)); cid += 1

    # ── Legend (top-left) ──
    lx, ly = 30, 810
    elements.append(_cell(pid, str(cid), "冻结模块",
        _box_style(FROZEN_FILL, FROZEN_STROKE, FROZEN_FONT, font_size=11),
        lx, ly, 120, 30)); cid += 1
    elements.append(_cell(pid, str(cid), "可训练模块",
        _box_style(TRAIN_FILL, TRAIN_STROKE, TRAIN_FONT, font_size=11),
        lx + 140, ly, 120, 30)); cid += 1
    elements.append(_cell(pid, str(cid), "选择性微调",
        _box_style(CFCA_FILL, CFCA_STROKE, CFCA_FONT, font_size=11, stroke_width=3),
        lx + 280, ly, 120, 30)); cid += 1
    elements.append(_cell(pid, str(cid), "诊断模块",
        _box_style(DIAG_FILL, DIAG_STROKE, DIAG_FONT, font_size=11),
        lx + 420, ly, 100, 30)); cid += 1

    # ── Section labels (top guidance) ──
    elements.append(_cell(pid, str(cid), "① 双模态输入",
        _box_style("none", "none", "#64748b", font_size=11, rounded=0),
        125, 55, 140, 25)); cid += 1
    elements.append(_cell(pid, str(cid), "② 输入端修正",
        _box_style("none", "none", "#64748b", font_size=11, rounded=0),
        495, 55, 140, 25)); cid += 1
    elements.append(_cell(pid, str(cid), "③ 冻结主干推理 + 协同修复",
        _box_style("none", "none", "#64748b", font_size=11, rounded=0),
        760, 55, 260, 25)); cid += 1
    elements.append(_cell(pid, str(cid), "④ 异常检测输出",
        _box_style("none", "none", "#64748b", font_size=11, rounded=0),
        1200, 55, 160, 25)); cid += 1

    # ── ① Inputs ──
    ix, iy = 40, 90
    rgb_in = str(cid)
    elements.append(_cell(pid, str(cid),
        "<b>Intensity</b><br>(RGB Image)",
        _box_style(INPUT_FILL, INPUT_STROKE, font_size=12),
        ix, iy, 180, 80)); cid += 1

    depth_in = str(cid)
    elements.append(_cell(pid, str(cid),
        "<b>Depth</b><br>(16-bit TIFF)",
        _box_style(INPUT_FILL, INPUT_STROKE, font_size=12),
        ix, iy + 230, 180, 80)); cid += 1

    # ── ② DepthAffinePEFT ──
    peft_in = str(cid)
    elements.append(_cell(pid, str(cid),
        "<b>DepthAffinePEFT</b><br><i>x</i><sub>d</sub>' = γ · <i>x</i><sub>d</sub> + β<br>"
        "<font style='font-size:10px;color:#2563eb'>(2 trainable params)</font>",
        _box_style(TRAIN_FILL, TRAIN_STROKE, font_size=11, stroke_width=2),
        340, iy + 230, 200, 80)); cid += 1

    # Arrow: RGB → backbone
    rgb_to_trd = str(cid)
    elements.append(_edge(pid, str(cid), rgb_in, "bb", _arrow_style(ARROW))); cid += 1

    # Arrow: Depth → PEFT
    depth_to_peft = str(cid)
    elements.append(_edge(pid, str(cid), depth_in, peft_in,
        _arrow_style(ARROW))); cid += 1

    # ── ③ TRD Backbone (large enclosure) ──
    bb_x, bb_y, bb_w, bb_h = 630, 85, 500, 400
    bb_id = str(cid)
    elements.append(_cell(pid, str(cid),
        "<b>Frozen TRD Backbone</b><br>"
        "<font style='font-size:10px'>Teacher-Student Distillation + Cross-modal Fusion</font>",
        _box_style(FROZEN_FILL, FROZEN_STROKE, FROZEN_FONT, font_size=13, stroke_width=3),
        bb_x, bb_y, bb_w, bb_h)); cid += 1

    # Sub-modules inside backbone
    sub_w, sub_h = 200, 80
    gap_x, gap_y = 40, 20
    sub_x0 = bb_x + 20
    sub_ry0 = bb_y + 60

    # Row 1: Teacher Enc + Student Dec (frozen, in sub-pale green)
    te_id = str(cid)
    elements.append(_cell(bb_id, str(cid),
        "<b>Teacher Encoder</b><br>"
        "<font style='font-size:9px'>WRN50_2 (frozen)</font>",
        _box_style(SUB_FILL, FROZEN_STROKE, FROZEN_FONT, font_size=11, stroke_width=1.5),
        sub_x0, sub_ry0, sub_w, sub_h)); cid += 1

    sd_id = str(cid)
    elements.append(_cell(bb_id, str(cid),
        "<b>Student Decoder</b><br>"
        "<font style='font-size:9px'>(frozen)</font>",
        _box_style(SUB_FILL, FROZEN_STROKE, FROZEN_FONT, font_size=11, stroke_width=1.5),
        sub_x0 + sub_w + gap_x, sub_ry0, sub_w, sub_h)); cid += 1

    # Row 2: CF (bottleneck) + CA (skip fusion) — selective FT
    cf_id = str(cid)
    elements.append(_cell(bb_id, str(cid),
        "<b>CF Module</b><br>"
        "<font style='font-size:9px;color:#0369a1'>Crossmodal Filter (Selective FT)</font>",
        _box_style(CFCA_FILL, CFCA_STROKE, CFCA_FONT, font_size=11, stroke_width=3),
        sub_x0, sub_ry0 + sub_h + gap_y, sub_w, sub_h)); cid += 1

    ca_id = str(cid)
    elements.append(_cell(bb_id, str(cid),
        "<b>CA Module</b><br>"
        "<font style='font-size:9px;color:#0369a1'>Crossmodal Amplifier (Selective FT)</font>",
        _box_style(CFCA_FILL, CFCA_STROKE, CFCA_FONT, font_size=11, stroke_width=3),
        sub_x0 + sub_w + gap_x, sub_ry0 + sub_h + gap_y, sub_w, sub_h)); cid += 1

    # Arrow: PEFT → backbone
    peft_to_bb = str(cid)
    elements.append(_edge(pid, str(cid), peft_in, bb_id,
        _arrow_style(TRAIN_STROKE),
        value="")); cid += 1

    # ── ④ Output: Fusion Anomaly Map ──
    out_id = str(cid)
    elements.append(_cell(pid, str(cid),
        "<b>Fusion Anomaly Map</b><br>"
        "<i>M</i> = <i>M</i><sub>i</sub><sup>norm</sup> + <i>M</i><sub>d</sub><sup>norm</sup>",
        _box_style(OUTPUT_FILL, OUTPUT_STROKE, OUTPUT_FONT, font_size=12, stroke_width=2),
        1240, iy + 150, 260, 100)); cid += 1

    # Arrow: backbone → output
    bb_to_out = str(cid)
    elements.append(_edge(pid, str(cid), bb_id, out_id,
        _arrow_style(ARROW))); cid += 1

    # ── Isolated AUROC diagnostic (top bar spanning backbone area) ──
    diag_id = str(cid)
    elements.append(_cell(pid, str(cid),
        "<b>isolated AUROC 诊断</b>:  "
        "Δ = AUROC<sub>cross</sub> − AUROC<sub>isolated</sub>  "
        "→ 量化跨模态路径贡献度",
        _box_style(DIAG_FILL, DIAG_STROKE, DIAG_FONT, font_size=12, stroke_width=2, dashed=1),
        630, 500, 500, 35)); cid += 1

    # Bidirectional arrow between backbone and diagnostic
    diag_edge = str(cid)
    elements.append(_edge(pid, str(cid), bb_id, diag_id,
        _arrow_style(DIAG_STROKE, width=1.5, end_arrow="classic"),
        value="")); cid += 1

    # ── Parameter budget box (right side) ──
    param_text = (
        "<b>训练参数统计</b><br>"
        "• DepthAffinePEFT: γ, β  (2 个)<br>"
        "• CF/CA Selective FT: ≈ 108 个<br>"
        "• 主干参数: 0 (全部冻结)<br>"
        "• 总计可训练: ≈ 110 个<br>"
        "<font style='font-size:9px'>占主干 ≈ 10<sup>−6</sup></font>"
    )
    param_id = str(cid)
    elements.append(_cell(pid, str(cid), param_text,
        _box_style(PARAM_FILL, PARAM_STROKE, "#4c1d95", font_size=11, stroke_width=2),
        1240, 500, 260, 150)); cid += 1

    # ── CAD patch management (bottom bar) ──
    cad_text = (
        "<b>CAD 补丁式持续适配管理</b>:  "
        "每个任务 T<sub>k</sub> 生成独立补丁 P<sub>k</sub> = {φ<sub>affine</sub><sup>k</sup>, "
        "φ<sub>cf/ca</sub><sup>k</sup>, m<sub>k</sub>}  "
        "→ 补丁仓库, 可独立加载 / 卸载 / 回滚"
    )
    cad_id = str(cid)
    elements.append(_cell(pid, str(cid), cad_text,
        _box_style("#f8fafc", "#94a3b8", "#475569", font_size=11, dashed=1),
        40, 580, 1460, 50)); cid += 1

    # ── Flow summary (bottom) ──
    flow_y = 660
    flow_items = [
        ("输入端修正", TRAIN_FONT, 40),
        ("→", ARROW, None),
        ("冻结主干推理", FROZEN_FONT, None),
        ("→", ARROW, None),
        ("协同选择性微调", CFCA_FONT, None),
        ("→", ARROW, None),
        ("融合异常图", OUTPUT_FONT, None),
        ("→", ARROW, None),
        ("isolated AUROC 诊断反馈", DIAG_FONT, None),
    ]
    fx = 40
    for text, color, w in flow_items:
        if text == "→":
            elements.append(_cell(pid, str(cid), "",
                _box_style("none", "none", font_size=14, rounded=0),
                fx, flow_y, 30, 20)); cid += 1
            fx += 30
        else:
            est_w = w or (len(text) * 16 + 20)
            elements.append(_cell(pid, str(cid),
                f"<font color='{color}'><b>{text}</b></font>",
                _box_style("none", "none", color, font_size=11, rounded=0),
                fx, flow_y, est_w, 25)); cid += 1
            fx += est_w + 5

    # ── Task-flow connector (bottom-left) ──
    task_text = (
        "<b>Task Stream</b><br>"
        "T<sub>0</sub> → T<sub>1</sub> → T<sub>2</sub> → ...<br>"
        "<font style='font-size:9px'>by view / time-window / condition</font>"
    )
    elements.append(_cell(pid, str(cid), task_text,
        _box_style("#f1f5f9", "#64748b", "#334155", font_size=10),
        40, 720, 200, 70)); cid += 1

    return elements


# ═══════════════════════════════════════════════════════════════════
# Figure 3-2: Detail mechanism diagram
# ═══════════════════════════════════════════════════════════════════

def build_fig3_2_elements(root_id: str) -> list[ET.Element]:
    elements: list[ET.Element] = []
    pid = root_id
    cid = 2

    # ── Title ──
    elements.append(_cell(pid, str(cid),
        "图 3-2  PEFT + CF/CA 协同选择性微调细节机制图",
        _box_style("none", "none", "#0f172a", font_size=18, rounded=0),
        350, 8, 900, 40)); cid += 1

    # ── Layout constants ──
    # Left side: two parallel branches (I-branch top, D-branch bottom)
    # Center-top: Teacher encoder (frozen per branch)
    # Center-middle: Bottleneck with CF
    # Center-bottom: Student decoder with CA at skip connections
    # Right: Anomaly maps → Fusion → Output

    branch_x = 30
    i_branch_y = 80
    d_branch_y = 420
    box_w = 180
    box_h = 65
    small_w = 140
    small_h = 55
    enc_x = 290
    bottle_x = 520
    dec_x = 750
    out_x = 1000
    fmap_w = 200
    fmap_h = 90

    # ═══════════════════════════════════════
    # Intensity Branch (top)
    # ═══════════════════════════════════════

    # I Input
    i_in = str(cid)
    elements.append(_cell(pid, str(cid),
        "<b>Intensity</b><br><font style='font-size:9px'>RGB Image (x<sub>i</sub>)</font>",
        _box_style(INPUT_FILL, INPUT_STROKE, font_size=12),
        branch_x, i_branch_y, box_w, box_h)); cid += 1

    # I Encoder (frozen)
    i_enc = str(cid)
    elements.append(_cell(pid, str(cid),
        "<b>Teacher Encoder</b><br><font style='font-size:9px'>WRN50_2 (frozen)<br>"
        "multi-scale features f<sub>i</sub><sup>l</sup></font>",
        _box_style(FROZEN_FILL, FROZEN_STROKE, FROZEN_FONT, font_size=11),
        enc_x, i_branch_y, box_w + 20, box_h)); cid += 1

    # I Enc → bottleneck
    i_enc_to_bn = str(cid)
    elements.append(_edge(pid, str(cid), i_in, i_enc,
        _arrow_style(ARROW))); cid += 1

    # I Bottleneck (frozen)
    i_bn = str(cid)
    elements.append(_cell(pid, str(cid),
        "<b>I-Bottleneck</b><br><font style='font-size:9px'>compressed rep (frozen)</font>",
        _box_style(FROZEN_FILL, FROZEN_STROKE, FROZEN_FONT, font_size=11),
        bottle_x, i_branch_y + 90, small_w, small_h)); cid += 1

    i_enc_to_bn_edge = str(cid)
    elements.append(_edge(pid, str(cid), i_enc, i_bn,
        _arrow_style(ARROW))); cid += 1

    # I Student Decoder (frozen)
    i_dec = str(cid)
    elements.append(_cell(pid, str(cid),
        "<b>Student Decoder</b><br><font style='font-size:9px'>(frozen)<br>"
        "recover teacher features</font>",
        _box_style(FROZEN_FILL, FROZEN_STROKE, FROZEN_FONT, font_size=11),
        dec_x, i_branch_y + 90, box_w, box_h)); cid += 1

    i_bn_to_dec = str(cid)
    elements.append(_edge(pid, str(cid), i_bn, i_dec,
        _arrow_style(ARROW))); cid += 1

    # I Anomaly Map
    i_amap = str(cid)
    elements.append(_cell(pid, str(cid),
        "<b>I-Anomaly Map M<sub>i</sub></b><br>"
        "<font style='font-size:9px'>teacher−student diff</font>",
        _box_style(OUTPUT_FILL, OUTPUT_STROKE, OUTPUT_FONT, font_size=11),
        out_x, i_branch_y + 90, fmap_w, fmap_h)); cid += 1
    i_dec_to_amap = str(cid)
    elements.append(_edge(pid, str(cid), i_dec, i_amap,
        _arrow_style(ARROW))); cid += 1

    # ═══════════════════════════════════════
    # Depth Branch (bottom)
    # ═══════════════════════════════════════

    # D Input (with PEFT)
    d_in = str(cid)
    elements.append(_cell(pid, str(cid),
        "<b>Depth</b><br><font style='font-size:9px'>16-bit TIFF (x<sub>d</sub>)</font>",
        _box_style(INPUT_FILL, INPUT_STROKE, font_size=12),
        branch_x, d_branch_y, box_w, box_h)); cid += 1

    # DepthAffinePEFT (trainable)
    peft_id = str(cid)
    elements.append(_cell(pid, str(cid),
        "<b>DepthAffinePEFT</b><br>"
        "x<sub>d</sub>' = <font color='#2563eb'>γ</font> · x<sub>d</sub> + "
        "<font color='#2563eb'>β</font><br>"
        "<font style='font-size:9px;color:#2563eb'>(2 trainable scalars)</font>",
        _box_style(TRAIN_FILL, TRAIN_STROKE, TRAIN_FONT, font_size=11, stroke_width=3),
        branch_x, d_branch_y + 85, box_w, box_h)); cid += 1

    d_in_to_peft = str(cid)
    elements.append(_edge(pid, str(cid), d_in, peft_id,
        _arrow_style(TRAIN_STROKE, width=2))); cid += 1

    # D Encoder (frozen)
    d_enc = str(cid)
    elements.append(_cell(pid, str(cid),
        "<b>Teacher Encoder</b><br><font style='font-size:9px'>WRN50_2 (frozen)<br>"
        "multi-scale features f<sub>d</sub><sup>l</sup></font>",
        _box_style(FROZEN_FILL, FROZEN_STROKE, FROZEN_FONT, font_size=11),
        enc_x, d_branch_y + 85, box_w + 20, box_h)); cid += 1

    peft_to_denc = str(cid)
    elements.append(_edge(pid, str(cid), peft_id, d_enc,
        _arrow_style(TRAIN_STROKE, width=2))); cid += 1

    # D Bottleneck (frozen)
    d_bn = str(cid)
    elements.append(_cell(pid, str(cid),
        "<b>D-Bottleneck</b><br><font style='font-size:9px'>compressed rep (frozen)</font>",
        _box_style(FROZEN_FILL, FROZEN_STROKE, FROZEN_FONT, font_size=11),
        bottle_x, d_branch_y + 175, small_w, small_h)); cid += 1

    d_enc_to_dbn = str(cid)
    elements.append(_edge(pid, str(cid), d_enc, d_bn,
        _arrow_style(ARROW))); cid += 1

    # D Student Decoder (frozen)
    d_dec = str(cid)
    elements.append(_cell(pid, str(cid),
        "<b>Student Decoder</b><br><font style='font-size:9px'>(frozen)<br>"
        "recover teacher features</font>",
        _box_style(FROZEN_FILL, FROZEN_STROKE, FROZEN_FONT, font_size=11),
        dec_x, d_branch_y + 175, box_w, box_h)); cid += 1

    d_bn_to_ddec = str(cid)
    elements.append(_edge(pid, str(cid), d_bn, d_dec,
        _arrow_style(ARROW))); cid += 1

    # D Anomaly Map
    d_amap = str(cid)
    elements.append(_cell(pid, str(cid),
        "<b>D-Anomaly Map M<sub>d</sub></b><br>"
        "<font style='font-size:9px'>teacher−student diff</font>",
        _box_style(OUTPUT_FILL, OUTPUT_STROKE, OUTPUT_FONT, font_size=11),
        out_x, d_branch_y + 175, fmap_w, fmap_h)); cid += 1
    d_dec_to_damap = str(cid)
    elements.append(_edge(pid, str(cid), d_dec, d_amap,
        _arrow_style(ARROW))); cid += 1

    # ═══════════════════════════════════════
    # CF Module (between bottlenecks, selective FT)
    # ═══════════════════════════════════════
    cf_x = bottle_x + small_w + 30
    cf_y = (i_branch_y + 90 + d_branch_y + 175 + small_h) // 2 - 45

    cf_mod = str(cid)
    elements.append(_cell(pid, str(cid),
        "<b>CF Module</b><br>"
        "<font style='font-size:9px;color:#0369a1'>Crossmodal Filter<br>"
        "at bottleneck<br>"
        "(Selective FT ≈ 54 params)</font>",
        _box_style(CFCA_FILL, CFCA_STROKE, CFCA_FONT, font_size=11, stroke_width=3),
        cf_x, cf_y, 160, 90)); cid += 1

    # CF bidirectional arrows from both bottlenecks
    cf_i = str(cid)
    elements.append(_edge(pid, str(cid), i_bn, cf_mod,
        _arrow_style(CFCA_STROKE, width=2, curved=1),
        value="I→D")); cid += 1
    cf_d = str(cid)
    elements.append(_edge(pid, str(cid), d_bn, cf_mod,
        _arrow_style(CFCA_STROKE, width=2, curved=1),
        value="D→I")); cid += 1

    # ═══════════════════════════════════════
    # CA Module (between decoder skip connections, selective FT)
    # ═══════════════════════════════════════
    ca_x = dec_x + box_w + 15
    ca_y = cf_y

    ca_mod = str(cid)
    elements.append(_cell(pid, str(cid),
        "<b>CA Module</b><br>"
        "<font style='font-size:9px;color:#0369a1'>Crossmodal Amplifier<br>"
        "at decoder skip fusion<br>"
        "(Selective FT ≈ 54 params)</font>",
        _box_style(CFCA_FILL, CFCA_STROKE, CFCA_FONT, font_size=11, stroke_width=3),
        ca_x, ca_y, 170, 90)); cid += 1

    # CA bidirectional arrows
    ca_i = str(cid)
    elements.append(_edge(pid, str(cid), i_dec, ca_mod,
        _arrow_style(CFCA_STROKE, width=2, curved=1),
        value="")); cid += 1
    ca_d = str(cid)
    elements.append(_edge(pid, str(cid), d_dec, ca_mod,
        _arrow_style(CFCA_STROKE, width=2, curved=1),
        value="")); cid += 1

    # ═══════════════════════════════════════
    # Fusion output (right of anomaly maps)
    # ═══════════════════════════════════════
    fusion_y = (i_branch_y + 90 + d_branch_y + 175 + fmap_h) // 2 - 45
    fusion_id = str(cid)
    elements.append(_cell(pid, str(cid),
        "<b>Fusion Anomaly Map</b><br>"
        "M = M<sub>i</sub><sup>norm</sup> + M<sub>d</sub><sup>norm</sup><br>"
        "<font style='font-size:9px'>normalized sum fusion</font>",
        _box_style(OUTPUT_FILL, OUTPUT_STROKE, OUTPUT_FONT, font_size=11, stroke_width=2),
        out_x + fmap_w + 50, fusion_y, 220, 100)); cid += 1

    # Arrows from anomaly maps to fusion
    i_amap_to_fusion = str(cid)
    elements.append(_edge(pid, str(cid), i_amap, fusion_id,
        _arrow_style(ARROW))); cid += 1
    d_amap_to_fusion = str(cid)
    elements.append(_edge(pid, str(cid), d_amap, fusion_id,
        _arrow_style(ARROW))); cid += 1

    # ═══════════════════════════════════════
    # Training loss annotations (bottom-left area)
    # ═══════════════════════════════════════
    loss_x = 30
    loss_y = 700
    loss_id = str(cid)
    elements.append(_cell(pid, str(cid),
        "<b>Training Loss</b><br>"
        "L = L<sub>FDM</sub> + λ<sub>cf</sub> · L<sub>CF</sub> "
        "+ λ<sub>ca</sub> · L<sub>CA</sub><br>"
        "<font style='font-size:9px'>L<sub>FDM</sub>: feature distribution matching (μ,σ)<br>"
        "L<sub>CF</sub>: crossmodal filter consistency<br>"
        "L<sub>CA</sub>: crossmodal amplifier consistency</font>",
        _box_style("#fefce8", "#ca8a04", "#854d0e", font_size=11, rounded=1, stroke_width=2),
        loss_x, loss_y, 420, 95)); cid += 1

    # ═══════════════════════════════════════
    # Inference annotations (bottom-right area)
    # ═══════════════════════════════════════
    inf_x = 500
    inf_id = str(cid)
    elements.append(_cell(pid, str(cid),
        "<b>Inference (no gradient)</b><br>"
        "1. x<sub>d</sub>' = γ · x<sub>d</sub> + β<br>"
        "2. Extract teacher features f<sub>i</sub>, f<sub>d</sub><br>"
        "3. CF/CA cross-modal modulation<br>"
        "4. Student recovery → M<sub>i</sub>, M<sub>d</sub><br>"
        "5. M = M<sub>i</sub><sup>norm</sup> + M<sub>d</sub><sup>norm</sup>",
        _box_style("#f0fdf4", "#16a34a", "#166534", font_size=11, rounded=1, stroke_width=2),
        inf_x, loss_y, 420, 95)); cid += 1

    # ═══════════════════════════════════════
    # Legend (bottom-right)
    # ═══════════════════════════════════════
    leg_x = 980
    leg_y = 700
    leg_id = str(cid)
    elements.append(_cell(pid, str(cid),
        "<b>Legend</b><br>"
        "<font color='#059669'>■ Frozen (no gradient)</font><br>"
        "<font color='#2563eb'>■ Trainable (PEFT params)</font><br>"
        "<font color='#0284c7'>■ Selective Fine-Tuning</font><br>"
        "<font color='#334155'>— Data flow</font><br>"
        "<font color='#ca8a04'>— Training signal</font>",
        _box_style("#f8fafc", "#cbd5e1", "#475569", font_size=10, rounded=1, stroke_width=1),
        leg_x, leg_y, 230, 140)); cid += 1

    # ═══════════════════════════════════════
    # CF/CA annotation tags
    # ═══════════════════════════════════════
    tag_style = _box_style(CFCA_FILL, CFCA_STROKE, CFCA_FONT, font_size=10, stroke_width=2)
    elements.append(_cell(pid, str(cid),
        "<b>Bottleneck</b><br>cross-modal<br>feature filter",
        tag_style, bottle_x - 30, i_branch_y + 90 - 45, 120, 50)); cid += 1
    elements.append(_cell(pid, str(cid),
        "<b>Decoder Skip</b><br>cross-modal<br>amplification",
        tag_style, dec_x - 30, i_branch_y + 90 - 45, 120, 50)); cid += 1

    # ═══════════════════════════════════════
    # Branch labels
    # ═══════════════════════════════════════
    label_style = _box_style("none", "none", "#64748b", font_size=13, rounded=0)
    elements.append(_cell(pid, str(cid), "Intensity Branch",
        label_style, branch_x - 10, i_branch_y - 30, 150, 25)); cid += 1
    elements.append(_cell(pid, str(cid), "Depth Branch (with PEFT)",
        label_style, branch_x - 10, d_branch_y - 30, 180, 25)); cid += 1

    return elements


# ═══════════════════════════════════════════════════════════════════
# Write .drawio files
# ═══════════════════════════════════════════════════════════════════

def _indent(elem: ET.Element, level: int = 0) -> None:
    """Minimal pretty-printer."""
    i = "\n" + "  " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for child in elem:
            _indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def write_drawio(name: str, builder_fn, page_id: str = "page-1") -> Path:
    diag = _make_diagram(name, page_id)
    model = diag.find("mxGraphModel")
    root_elem = model.find("root")
    elements = builder_fn("1")
    for el in elements:
        root_elem.append(el)
    mxfile = _make_mxfile([diag])
    _indent(mxfile)
    tree = ET.ElementTree(mxfile)
    out_path = FIG_DIR / f"{name}.drawio"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(out_path), encoding="UTF-8", xml_declaration=True)
    print(f"  Wrote {out_path}")
    return out_path


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    write_drawio("fig3_1_method_pipeline", build_fig3_1_elements)
    write_drawio("fig3_2_peft_cfca_detail", build_fig3_2_elements)


if __name__ == "__main__":
    main()
