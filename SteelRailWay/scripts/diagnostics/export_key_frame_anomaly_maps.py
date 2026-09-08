#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Export side-by-side anomaly-map visualizations for key frames."""

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
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from datasets.rail_dataset import RailDualModalDataset
from eval.eval_utils import cal_anomaly_map
from models.trd.decoder import ResNet50DualModalDecoder
from models.trd.encoder import ResNet50Encoder
from scripts.eval.eval_from_ckpt import (
    SCORE_SOURCES,
    SOURCE_SEMANTICS,
    amp_context_factory,
    build_assist_feature_list,
    load_depth_peft,
    load_feature_mean_payload,
    resolve_amp_dtype,
    safe_load_ckpt,
    strip_prefix,
)


@dataclass
class PatchBox:
    y0: int
    y1: int
    x0: int
    x1: int


@dataclass
class FrameBundle:
    frame_id: str
    view_id: int
    label: int
    label_name: str
    rgb_path: str
    depth_path: str
    gt_path: str | None
    rgb_visible: np.ndarray
    depth_visible: np.ndarray
    gt_visible: np.ndarray
    visible_x0: int
    visible_x1: int
    patch_boxes: list[PatchBox]
    rgb_batch: torch.Tensor
    depth_batch: torch.Tensor
    image_height: int
    visible_width: int


def build_parser():
    parser = argparse.ArgumentParser(
        description="Export side-by-side anomaly-map figures for selected key frames."
    )
    parser.add_argument(
        "--key_manifest",
        type=str,
        default="outputs/rail_peft/cam4_p1_20260501_225618/diagnostics/key_frames_server/manifest_all.csv",
        help="关键帧清单 CSV（由 export_key_frames.py 生成）",
    )
    parser.add_argument(
        "--rank_summary",
        type=str,
        default="outputs/rail_peft/cam4_p1_20260501_225618/diagnostics/branch_rank_analysis_server/summary.csv",
        help="rank analysis 的 summary.csv，内含 comparison -> source/config 映射",
    )
    parser.add_argument(
        "--scores_root",
        type=str,
        default="outputs/rail_peft/cam4_p1_20260501_225618/diagnostics/branch_isolation_server",
        help="包含 cam*_baseline/result.json 与 cam*_with_cam4peft/result.json 的目录",
    )
    parser.add_argument(
        "--test_root",
        type=str,
        default="rail_mvtec_gt_test",
        help="本地测试集根目录",
    )
    parser.add_argument(
        "--train_root",
        type=str,
        default="data_20260327",
        help="仅在 assist stats 缓存缺失时才需要；本脚本默认优先复用已有缓存",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="输出目录，默认写到 key_frames 同级的 key_frame_maps[_server]",
    )
    parser.add_argument(
        "--comparisons",
        type=str,
        nargs="*",
        default=None,
        help="仅处理指定 comparison 名称",
    )
    parser.add_argument(
        "--frame_ids",
        type=str,
        nargs="*",
        default=None,
        help="仅处理指定 frame_id",
    )
    parser.add_argument(
        "--limit_frames",
        type=int,
        default=None,
        help="调试用：最多处理前 N 个 manifest 条目",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=512,
        help="模型输入尺寸，应与训练/评估保持一致",
    )
    parser.add_argument(
        "--depth_norm",
        type=str,
        default="zscore",
        help="Depth 归一化方式",
    )
    parser.add_argument(
        "--use_patch",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no_patch",
        action="store_false",
        dest="use_patch",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=900,
    )
    parser.add_argument(
        "--patch_stride",
        type=int,
        default=850,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
    )
    parser.add_argument(
        "--channels_last",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no_channels_last",
        action="store_false",
        dest="channels_last",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="导出 figure 的 DPI",
    )
    parser.add_argument(
        "--diff_cmap",
        type=str,
        default="coolwarm",
    )
    parser.add_argument(
        "--allow_name_fallback",
        action="store_true",
        help="找不到 result.json 中的精确 artifact 路径时，允许按文件名在邻近目录搜索；默认关闭，避免误用不同 checkpoint",
    )
    parser.add_argument(
        "--depth_pct_low",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--depth_pct_high",
        type=float,
        default=99.0,
    )
    return parser


def default_out_dir(key_manifest: Path) -> Path:
    parent = key_manifest.parent
    name = "key_frame_maps_server" if parent.name.endswith("_server") else "key_frame_maps"
    return parent.parent / name


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_artifact_path(path_str: str, allow_name_fallback: bool = False) -> str:
    if not path_str:
        return ""
    raw = Path(path_str)
    candidates = [raw]
    if not raw.is_absolute():
        candidates.append(_PROJ_ROOT / raw)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    relocated_candidates = []
    for candidate in candidates:
        relocated_candidates.extend([
            candidate.parent / "final" / candidate.name,
            candidate.parent / "checkpoints" / candidate.name,
        ])
    for candidate in relocated_candidates:
        if candidate.exists():
            return str(candidate)

    if not allow_name_fallback:
        raise FileNotFoundError(
            f"Artifact path does not exist: {path_str}. "
            "Copy the exact checkpoint/cache locally or rerun on the server. "
            "For diagnostic-only fallback, pass --allow_name_fallback."
        )

    search_roots = []
    for candidate in candidates:
        search_roots.append(candidate.parent)
        search_roots.append(candidate.parent.parent)

    seen = set()
    for root in search_roots:
        if not root or str(root) in seen or not root.exists():
            continue
        seen.add(str(root))
        matches = list(root.rglob(raw.name))
        if len(matches) == 1:
            return str(matches[0])
        if len(matches) > 1:
            return str(matches[0])
    raise FileNotFoundError(f"Artifact path does not exist and no fallback match was found: {path_str}")


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_rank_specs(path: Path) -> dict[str, dict[str, str]]:
    rows = load_csv_rows(path)
    return {row["name"]: row for row in rows}


def filter_manifest_rows(rows, comparisons=None, frame_ids=None, limit_frames=None):
    filtered = []
    allowed_comparisons = set(comparisons or [])
    allowed_frame_ids = set(frame_ids or [])
    for row in rows:
        if allowed_comparisons and row["comparison"] not in allowed_comparisons:
            continue
        if allowed_frame_ids and row["frame_id"] not in allowed_frame_ids:
            continue
        filtered.append(row)
    if limit_frames is not None:
        filtered = filtered[: max(0, int(limit_frames))]
    return filtered


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def depth_preview(depth_array: np.ndarray, pct_low: float, pct_high: float) -> np.ndarray:
    depth = depth_array.astype(np.float32)
    valid = np.isfinite(depth)
    if not np.any(valid):
        return np.zeros(depth.shape + (3,), dtype=np.uint8)
    values = depth[valid]
    low = float(np.percentile(values, pct_low))
    high = float(np.percentile(values, pct_high))
    if high <= low:
        low = float(values.min())
        high = float(values.max())
    if high <= low:
        high = low + 1.0
    norm = np.clip((depth - low) / (high - low), 0.0, 1.0)
    preview = (norm * 255.0).astype(np.uint8)
    return cv2.cvtColor(preview, cv2.COLOR_GRAY2RGB)


def crop_visible_strip(image: np.ndarray, patch_size: int) -> tuple[np.ndarray, int, int]:
    height, width = image.shape[:2]
    if width > patch_size:
        x0 = (width - patch_size) // 2
        x1 = x0 + patch_size
    else:
        x0 = 0
        x1 = width
    if image.ndim == 3:
        return image[:, x0:x1, :].copy(), x0, x1
    return image[:, x0:x1].copy(), x0, x1


def patch_box_for_index(img_h: int, visible_w: int, patch_size: int, patch_stride: int, patch_idx: int) -> PatchBox:
    y0 = patch_idx * patch_stride
    y1 = y0 + patch_size
    if y1 > img_h:
        y1 = img_h
        y0 = y1 - patch_size
    return PatchBox(y0=y0, y1=y1, x0=0, x1=visible_w)


def overlay_heatmap(rgb_image: np.ndarray, amap: np.ndarray, cmap_name="jet", alpha=0.45) -> np.ndarray:
    rgb = rgb_image.astype(np.float32) / 255.0
    amap = amap.astype(np.float32)
    if amap.size == 0:
        return rgb
    vmin = float(amap.min())
    vmax = float(amap.max())
    if vmax <= vmin:
        norm = np.zeros_like(amap, dtype=np.float32)
    else:
        norm = (amap - vmin) / (vmax - vmin)
    cmap = plt.get_cmap(cmap_name)
    color = cmap(norm)[..., :3].astype(np.float32)
    blend = (1.0 - alpha) * rgb + alpha * color
    return np.clip(blend, 0.0, 1.0)


def overlay_mask(rgb_image: np.ndarray, gt_mask: np.ndarray, alpha=0.55) -> np.ndarray:
    rgb = rgb_image.astype(np.float32) / 255.0
    if gt_mask is None or gt_mask.size == 0 or float(gt_mask.max()) <= 0:
        return rgb
    mask = (gt_mask > 0).astype(np.float32)
    color = np.zeros_like(rgb, dtype=np.float32)
    color[..., 0] = 1.0
    blend = rgb.copy()
    blend = blend * (1.0 - alpha * mask[..., None]) + color * (alpha * mask[..., None])
    return np.clip(blend, 0.0, 1.0)


def render_diff_map(delta_map: np.ndarray, cmap_name="coolwarm") -> np.ndarray:
    delta = delta_map.astype(np.float32)
    vmax = float(np.max(np.abs(delta))) if delta.size else 0.0
    if vmax <= 0:
        norm = np.full(delta.shape, 0.5, dtype=np.float32)
    else:
        norm = np.clip((delta / (2.0 * vmax)) + 0.5, 0.0, 1.0)
    cmap = plt.get_cmap(cmap_name)
    return cmap(norm)[..., :3].astype(np.float32)


class DatasetCache:
    def __init__(self, args):
        self.args = args
        self.datasets: dict[int, RailDualModalDataset] = {}
        self.frame_index: dict[int, dict[str, int]] = {}

    def get_dataset(self, view_id: int) -> RailDualModalDataset:
        if view_id not in self.datasets:
            dataset = RailDualModalDataset(
                train_root=self.args.train_root,
                test_root=self.args.test_root,
                view_id=view_id,
                split="test",
                img_size=self.args.img_size,
                depth_norm=self.args.depth_norm,
                use_patch=self.args.use_patch,
                patch_size=self.args.patch_size,
                patch_stride=self.args.patch_stride,
                preload=False,
                preload_workers=0,
            )
            self.datasets[view_id] = dataset
            self.frame_index[view_id] = {
                sample["frame_id"]: idx for idx, sample in enumerate(dataset.samples)
            }
        return self.datasets[view_id]

    def build_frame_bundle(self, view_id: int, frame_id: str) -> FrameBundle:
        dataset = self.get_dataset(view_id)
        if frame_id not in self.frame_index[view_id]:
            raise KeyError(f"Frame not found in test split: Cam{view_id} {frame_id}")
        sample_idx = self.frame_index[view_id][frame_id]
        sample = dataset.samples[sample_idx]
        label = int(sample["label"])
        label_name = "good" if label == 0 else "broken"

        rgb_full = cv2.imread(sample["rgb_path"], cv2.IMREAD_COLOR)
        if rgb_full is None:
            raise FileNotFoundError(sample["rgb_path"])
        rgb_full = cv2.cvtColor(rgb_full, cv2.COLOR_BGR2RGB)
        depth_full = cv2.imread(sample["depth_path"], cv2.IMREAD_UNCHANGED)
        if depth_full is None:
            raise FileNotFoundError(sample["depth_path"])
        gt_full = None
        gt_path = sample.get("gt_path")
        if gt_path and os.path.exists(gt_path):
            gt_full = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt_full is None:
            gt_full = np.zeros(rgb_full.shape[:2], dtype=np.uint8)

        rgb_visible, x0, x1 = crop_visible_strip(rgb_full, dataset.patch_size)
        depth_visible_raw, _, _ = crop_visible_strip(depth_full, dataset.patch_size)
        gt_visible, _, _ = crop_visible_strip(gt_full, dataset.patch_size)

        rgb_tensors = []
        depth_tensors = []
        patch_boxes = []
        for patch_idx in range(dataset.num_patches):
            item = dataset[sample_idx * dataset.num_patches + patch_idx]
            rgb_tensors.append(item["intensity"])
            depth_tensors.append(item["depth"])
            patch_boxes.append(
                patch_box_for_index(
                    img_h=rgb_visible.shape[0],
                    visible_w=rgb_visible.shape[1],
                    patch_size=dataset.patch_size,
                    patch_stride=dataset.patch_stride,
                    patch_idx=patch_idx,
                )
            )

        rgb_batch = torch.stack(rgb_tensors, dim=0)
        depth_batch = torch.stack(depth_tensors, dim=0)
        return FrameBundle(
            frame_id=frame_id,
            view_id=view_id,
            label=label,
            label_name=label_name,
            rgb_path=sample["rgb_path"],
            depth_path=sample["depth_path"],
            gt_path=gt_path,
            rgb_visible=rgb_visible,
            depth_visible=depth_preview(depth_visible_raw, self.args.depth_pct_low, self.args.depth_pct_high),
            gt_visible=(gt_visible > 0).astype(np.uint8),
            visible_x0=x0,
            visible_x1=x1,
            patch_boxes=patch_boxes,
            rgb_batch=rgb_batch,
            depth_batch=depth_batch,
            image_height=rgb_visible.shape[0],
            visible_width=rgb_visible.shape[1],
        )


class FrameMapRunner:
    def __init__(self, config_name: str, result_payload: dict, device: torch.device, args):
        self.config_name = config_name
        self.result_payload = result_payload
        self.device = device
        self.args = args
        self.amp_dtype = resolve_amp_dtype(args.precision, device)
        self.map_cache: dict[str, dict[str, dict[str, object]]] = {}
        self._load_models()
        self._load_assist_feature_means()

    def _load_models(self):
        self.teacher_rgb = ResNet50Encoder(pretrained=True).to(self.device).eval()
        self.teacher_depth = ResNet50Encoder(pretrained=True).to(self.device).eval()
        for param in self.teacher_rgb.parameters():
            param.requires_grad_(False)
        for param in self.teacher_depth.parameters():
            param.requires_grad_(False)

        self.student_rgb = ResNet50DualModalDecoder(pretrained=False).to(self.device).eval()
        self.student_depth = ResNet50DualModalDecoder(pretrained=False).to(self.device).eval()

        if self.args.channels_last and self.device.type == "cuda":
            self.teacher_rgb = self.teacher_rgb.to(memory_format=torch.channels_last)
            self.teacher_depth = self.teacher_depth.to(memory_format=torch.channels_last)
            self.student_rgb = self.student_rgb.to(memory_format=torch.channels_last)
            self.student_depth = self.student_depth.to(memory_format=torch.channels_last)

        ckpt_path = resolve_artifact_path(
            self.result_payload["ckpt"],
            allow_name_fallback=self.args.allow_name_fallback,
        )
        ckpt = safe_load_ckpt(ckpt_path, self.device)
        self.student_rgb.load_state_dict(strip_prefix(ckpt["student_rgb"]))
        self.student_depth.load_state_dict(strip_prefix(ckpt["student_depth"]))
        self.result_payload["ckpt"] = ckpt_path

        if self.result_payload.get("depth_peft_ckpt"):
            peft_path = resolve_artifact_path(
                self.result_payload["depth_peft_ckpt"],
                allow_name_fallback=self.args.allow_name_fallback,
            )
            peft, _ = load_depth_peft(peft_path, self.device)
            self.teacher_depth = self.teacher_depth.to(self.device).eval()
            from rail_peft import DepthEncoderWithPEFT

            self.teacher_depth = DepthEncoderWithPEFT(self.teacher_depth, peft).to(self.device).eval()
            self.result_payload["depth_peft_ckpt"] = peft_path

    def _load_assist_feature_means(self):
        self.assist_feature_means = {}
        assist_fill = self.result_payload.get("assist_fill_mode", "train_mean")
        assist_stats_dir = resolve_artifact_path(
            self.result_payload.get("assist_stats_dir", ""),
            allow_name_fallback=self.args.allow_name_fallback,
        )
        if assist_fill != "train_mean" or not assist_stats_dir:
            return
        view_id = int(self.result_payload["view_id"])
        stats_dir = Path(assist_stats_dir)
        for modality in ("rgb", "depth"):
            path = stats_dir / f"cam{view_id}_{modality}_train_mean.pt"
            if path.exists():
                self.assist_feature_means[modality] = load_feature_mean_payload(path, self.device)

    @torch.no_grad()
    def compute_maps(self, bundle: FrameBundle) -> dict[str, dict[str, object]]:
        if bundle.frame_id in self.map_cache:
            return self.map_cache[bundle.frame_id]

        rgb = bundle.rgb_batch.to(self.device, non_blocking=True)
        depth = bundle.depth_batch.to(self.device, non_blocking=True)
        if self.args.channels_last and self.device.type == "cuda":
            rgb = rgb.contiguous(memory_format=torch.channels_last)
            depth = depth.contiguous(memory_format=torch.channels_last)

        with (amp_context_factory(self.device, self.amp_dtype)() if self.amp_dtype is not None else nullcontext()):
            feat_t_rgb = self.teacher_rgb(rgb)
            feat_t_depth = self.teacher_depth(depth)
            _, _, feat_s_rgb, _ = self.student_rgb(feat_t_rgb, feat_t_depth)
            _, _, feat_s_depth, _ = self.student_depth(feat_t_depth, feat_t_rgb)

            assist_fill = self.result_payload.get("assist_fill_mode", "train_mean")
            feat_t_depth_assist = build_assist_feature_list(
                "depth",
                feat_t_depth,
                assist_fill,
                self.assist_feature_means,
            )
            feat_t_rgb_assist = build_assist_feature_list(
                "rgb",
                feat_t_rgb,
                assist_fill,
                self.assist_feature_means,
            )
            _, _, feat_s_rgb_isolated, _ = self.student_rgb(feat_t_rgb, feat_t_depth_assist)
            _, _, feat_s_depth_isolated, _ = self.student_depth(feat_t_depth, feat_t_rgb_assist)

        amap_rgb, _ = cal_anomaly_map(feat_s_rgb, feat_t_rgb, out_size=(self.args.img_size, self.args.img_size), amap_mode="mul")
        amap_depth, _ = cal_anomaly_map(feat_s_depth, feat_t_depth, out_size=(self.args.img_size, self.args.img_size), amap_mode="mul")
        amap_rgb_isolated, _ = cal_anomaly_map(
            feat_s_rgb_isolated,
            feat_t_rgb,
            out_size=(self.args.img_size, self.args.img_size),
            amap_mode="mul",
        )
        amap_depth_isolated, _ = cal_anomaly_map(
            feat_s_depth_isolated,
            feat_t_depth,
            out_size=(self.args.img_size, self.args.img_size),
            amap_mode="mul",
        )

        patch_maps_by_source = {
            "rgb": np.asarray(amap_rgb, dtype=np.float32),
            "depth": np.asarray(amap_depth, dtype=np.float32),
            "fusion": np.asarray(amap_rgb, dtype=np.float32) + np.asarray(amap_depth, dtype=np.float32),
            "rgb_isolated": np.asarray(amap_rgb_isolated, dtype=np.float32),
            "depth_isolated": np.asarray(amap_depth_isolated, dtype=np.float32),
        }

        outputs: dict[str, dict[str, object]] = {}
        for source, patch_maps in patch_maps_by_source.items():
            if patch_maps.ndim == 2:
                patch_maps = patch_maps[None, ...]
            full_map = np.zeros((bundle.image_height, bundle.visible_width), dtype=np.float32)
            patch_scores = []
            for patch_idx, patch_map in enumerate(patch_maps):
                box = bundle.patch_boxes[patch_idx]
                box_h = box.y1 - box.y0
                box_w = box.x1 - box.x0
                patch_resized = cv2.resize(
                    patch_map.astype(np.float32),
                    (box_w, box_h),
                    interpolation=cv2.INTER_LINEAR,
                )
                full_map[box.y0:box.y1, box.x0:box.x1] = np.maximum(
                    full_map[box.y0:box.y1, box.x0:box.x1],
                    patch_resized,
                )
                patch_scores.append(float(patch_map.max()))
            top_patch_idx = int(np.argmax(patch_scores)) if patch_scores else 0
            outputs[source] = {
                "map": full_map,
                "patch_scores": patch_scores,
                "top_patch_idx": top_patch_idx,
                "top_patch_box": bundle.patch_boxes[top_patch_idx] if patch_scores else None,
            }

        self.map_cache[bundle.frame_id] = outputs
        return outputs


def make_figure(
    bundle: FrameBundle,
    source: str,
    source_semantics: str,
    row: dict[str, str],
    baseline_maps: dict[str, dict[str, object]],
    candidate_maps: dict[str, dict[str, object]],
    figure_path: Path,
    diff_cmap: str,
    dpi: int,
):
    baseline_map = baseline_maps[source]["map"]
    candidate_map = candidate_maps[source]["map"]
    delta_map = candidate_map - baseline_map

    map_min = float(min(baseline_map.min(), candidate_map.min()))
    map_max = float(max(baseline_map.max(), candidate_map.max()))
    if map_max <= map_min:
        map_min = 0.0
        map_max = 1.0

    baseline_overlay = overlay_heatmap(
        bundle.rgb_visible,
        (baseline_map - map_min) / (map_max - map_min + 1e-8),
    )
    candidate_overlay = overlay_heatmap(
        bundle.rgb_visible,
        (candidate_map - map_min) / (map_max - map_min + 1e-8),
    )
    gt_overlay = overlay_mask(bundle.rgb_visible, bundle.gt_visible)
    delta_panel = render_diff_map(delta_map, cmap_name=diff_cmap)

    aspect_ratio = bundle.image_height / max(1, bundle.visible_width)
    fig_height = max(8.0, min(14.0, 4.0 + aspect_ratio * 1.4))
    fig, axes = plt.subplots(1, 6, figsize=(20, fig_height), constrained_layout=True)

    panels = [
        ("RGB Visible Strip", bundle.rgb_visible.astype(np.float32) / 255.0),
        ("Depth Preview", bundle.depth_visible.astype(np.float32) / 255.0),
        ("GT / Label", gt_overlay),
        (
            f"Baseline {source}\nscore={safe_float(row['baseline_score']):.5f} rank={safe_int(row['baseline_rank'])}",
            baseline_overlay,
        ),
        (
            f"PEFT {source}\nscore={safe_float(row['candidate_score']):.5f} rank={safe_int(row['candidate_rank'])}",
            candidate_overlay,
        ),
        (
            f"Delta (PEFT-baseline)\nΔscore={safe_float(row['delta_score']):.5f} Δcontr={safe_float(row['delta_contribution']):.4f}",
            delta_panel,
        ),
    ]

    for ax, (title, image) in zip(axes, panels):
        ax.imshow(image, aspect="auto")
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    label_name = row.get("label_name", bundle.label_name)
    reasons = row.get("selection_reasons", "")
    fig.suptitle(
        f"{row['comparison']} | {bundle.frame_id} | label={label_name} | "
        f"net_flip={safe_int(row['net_flip_count'])} | reasons={reasons}\n"
        f"{source_semantics} | visible_x=[{bundle.visible_x0}, {bundle.visible_x1})",
        fontsize=12,
    )
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_metadata(
    path: Path,
    bundle: FrameBundle,
    row: dict[str, str],
    comparison_spec: dict[str, str],
    baseline_result: dict,
    candidate_result: dict,
    baseline_maps: dict[str, dict[str, object]],
    candidate_maps: dict[str, dict[str, object]],
    source: str,
):
    payload = {
        "comparison": row["comparison"],
        "frame_id": bundle.frame_id,
        "view_id": bundle.view_id,
        "label": bundle.label,
        "label_name": bundle.label_name,
        "source": source,
        "source_semantics": SOURCE_SEMANTICS.get(source, ""),
        "selection_reasons": row.get("selection_reasons", ""),
        "visible_x0": bundle.visible_x0,
        "visible_x1": bundle.visible_x1,
        "baseline_dir": comparison_spec["baseline_dir"],
        "candidate_dir": comparison_spec["candidate_dir"],
        "baseline_ckpt": baseline_result["ckpt"],
        "candidate_ckpt": candidate_result["ckpt"],
        "baseline_depth_peft_ckpt": baseline_result.get("depth_peft_ckpt", ""),
        "candidate_depth_peft_ckpt": candidate_result.get("depth_peft_ckpt", ""),
        "baseline_score": safe_float(row["baseline_score"]),
        "candidate_score": safe_float(row["candidate_score"]),
        "delta_score": safe_float(row["delta_score"]),
        "baseline_rank": safe_int(row["baseline_rank"]),
        "candidate_rank": safe_int(row["candidate_rank"]),
        "delta_rank": safe_int(row["delta_rank"]),
        "baseline_contribution": safe_float(row["baseline_contribution"]),
        "candidate_contribution": safe_float(row["candidate_contribution"]),
        "delta_contribution": safe_float(row["delta_contribution"]),
        "positive_flip_count": safe_int(row["positive_flip_count"]),
        "negative_flip_count": safe_int(row["negative_flip_count"]),
        "net_flip_count": safe_int(row["net_flip_count"]),
        "baseline_top_patch_idx": int(baseline_maps[source]["top_patch_idx"]),
        "candidate_top_patch_idx": int(candidate_maps[source]["top_patch_idx"]),
        "baseline_patch_scores": baseline_maps[source]["patch_scores"],
        "candidate_patch_scores": candidate_maps[source]["patch_scores"],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    args = build_parser().parse_args()
    key_manifest = Path(args.key_manifest)
    rank_summary = Path(args.rank_summary)
    scores_root = Path(args.scores_root)
    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir(key_manifest)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = load_csv_rows(key_manifest)
    manifest_rows = filter_manifest_rows(
        manifest_rows,
        comparisons=args.comparisons,
        frame_ids=args.frame_ids,
        limit_frames=args.limit_frames,
    )
    rank_specs = load_rank_specs(rank_summary)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Output dir: {out_dir}")
    print(f"Selected manifest rows: {len(manifest_rows)}")

    dataset_cache = DatasetCache(args)
    runner_cache: dict[str, FrameMapRunner] = {}
    result_rows: list[dict[str, object]] = []

    for index, row in enumerate(manifest_rows, start=1):
        comparison = row["comparison"]
        if comparison not in rank_specs:
            raise KeyError(f"Missing comparison spec in rank summary: {comparison}")
        spec = rank_specs[comparison]
        source = spec["source"]
        view_id = safe_int(row["view_id"])
        frame_id = row["frame_id"]

        print(
            f"[{index}/{len(manifest_rows)}] {comparison} | Cam{view_id} | {frame_id} | source={source}"
        )

        baseline_name = spec["baseline_dir"]
        candidate_name = spec["candidate_dir"]
        if baseline_name not in runner_cache:
            baseline_result = load_json(scores_root / baseline_name / "result.json")
            runner_cache[baseline_name] = FrameMapRunner(baseline_name, baseline_result, device, args)
        if candidate_name not in runner_cache:
            candidate_result = load_json(scores_root / candidate_name / "result.json")
            runner_cache[candidate_name] = FrameMapRunner(candidate_name, candidate_result, device, args)

        baseline_runner = runner_cache[baseline_name]
        candidate_runner = runner_cache[candidate_name]
        baseline_result = baseline_runner.result_payload
        candidate_result = candidate_runner.result_payload

        bundle = dataset_cache.build_frame_bundle(view_id, frame_id)
        baseline_maps = baseline_runner.compute_maps(bundle)
        candidate_maps = candidate_runner.compute_maps(bundle)

        frame_dir = out_dir / comparison / f"{bundle.label_name}_{frame_id}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        figure_path = frame_dir / "figure.png"
        metadata_path = frame_dir / "metadata.json"
        baseline_map_path = frame_dir / f"{source}_baseline.npy"
        candidate_map_path = frame_dir / f"{source}_peft.npy"
        delta_map_path = frame_dir / f"{source}_delta.npy"

        make_figure(
            bundle=bundle,
            source=source,
            source_semantics=SOURCE_SEMANTICS.get(source, ""),
            row=row,
            baseline_maps=baseline_maps,
            candidate_maps=candidate_maps,
            figure_path=figure_path,
            diff_cmap=args.diff_cmap,
            dpi=args.dpi,
        )

        np.save(baseline_map_path, baseline_maps[source]["map"])
        np.save(candidate_map_path, candidate_maps[source]["map"])
        np.save(delta_map_path, candidate_maps[source]["map"] - baseline_maps[source]["map"])
        write_metadata(
            metadata_path,
            bundle,
            row,
            spec,
            baseline_result,
            candidate_result,
            baseline_maps,
            candidate_maps,
            source,
        )

        result_rows.append(
            {
                "comparison": comparison,
                "frame_id": frame_id,
                "view_id": view_id,
                "label_name": bundle.label_name,
                "source": source,
                "baseline_score": safe_float(row["baseline_score"]),
                "candidate_score": safe_float(row["candidate_score"]),
                "delta_score": safe_float(row["delta_score"]),
                "delta_contribution": safe_float(row["delta_contribution"]),
                "net_flip_count": safe_int(row["net_flip_count"]),
                "figure_path": str(figure_path),
                "metadata_path": str(metadata_path),
                "baseline_map_path": str(baseline_map_path),
                "candidate_map_path": str(candidate_map_path),
                "delta_map_path": str(delta_map_path),
            }
        )

    manifest_out = out_dir / "manifest.csv"
    with open(manifest_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "comparison",
                "frame_id",
                "view_id",
                "label_name",
                "source",
                "baseline_score",
                "candidate_score",
                "delta_score",
                "delta_contribution",
                "net_flip_count",
                "figure_path",
                "metadata_path",
                "baseline_map_path",
                "candidate_map_path",
                "delta_map_path",
            ],
        )
        writer.writeheader()
        for row in result_rows:
            writer.writerow(row)

    summary_out = out_dir / "summary.txt"
    with open(summary_out, "w", encoding="utf-8") as f:
        f.write("Key-frame anomaly-map export\n")
        f.write(f"created_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"key_manifest: {key_manifest}\n")
        f.write(f"rank_summary: {rank_summary}\n")
        f.write(f"scores_root: {scores_root}\n")
        f.write(f"test_root: {args.test_root}\n")
        f.write(f"device: {device}\n")
        f.write(f"precision: {args.precision}\n")
        f.write(f"num_figures: {len(result_rows)}\n\n")
        grouped = {}
        for row in result_rows:
            grouped.setdefault(row["comparison"], []).append(row)
        for comparison, rows in grouped.items():
            f.write(f"{comparison}: {len(rows)} frames\n")
            for row in rows:
                f.write(
                    f"  - {row['frame_id']} ({row['label_name']}): "
                    f"source={row['source']}, "
                    f"baseline={row['baseline_score']:.5f}, "
                    f"candidate={row['candidate_score']:.5f}, "
                    f"delta_contribution={row['delta_contribution']:.4f}, "
                    f"figure={row['figure_path']}\n"
                )
            f.write("\n")

    print(f"Figure manifest: {manifest_out}")
    print(f"Summary: {summary_out}")


if __name__ == "__main__":
    main()
