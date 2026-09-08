#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Validate Cam4 fusion imbalance mechanism for cf_ca_repair vs peft_full_then_cf_ca.

This script is intentionally independent from the main evaluator. It reuses the
existing model/dataset/anomaly-map stack when checkpoints are available, and
writes all outputs to a timestamped directory under ``results/``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

_PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from datasets.rail_dataset import RailDualModalDataset
from eval.eval_utils import cal_anomaly_map
from models.trd.decoder import ResNet50DualModalDecoder
from models.trd.encoder import ResNet50Encoder
from rail_peft import DepthEncoderWithPEFT
from scripts.eval.eval_from_ckpt import (
    amp_context_factory,
    load_depth_peft,
    load_depth_peft_from_joint_ckpt,
    resolve_amp_dtype,
    safe_load_ckpt,
    strip_prefix,
)


DEFAULT_CFCA_REPAIR_CKPT = (
    "outputs/rail_ablation/cam4_cfca_repair/cf_ca/cam4_cf_ca_20260504_170923/final/repair_cam4.pth"
)
DEFAULT_CFCA_REPAIR_PEFT = "outputs/rail_peft/cam4_p1_20260501_225618/final/final_peft_cam4.pth"
DEFAULT_CHAINED_CKPT = (
    "outputs/rail_ablation/cam4_peft_full_then_cf_ca/"
    "cam4_peft_full_then_cf_ca_20260504_200444/peft_full_then_cf_ca/final/peft_full_then_cf_ca_cam4.pth"
)
DEFAULT_RESULTS_ROOT = "results"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate Cam4 fusion imbalance by comparing scale and fusion rules."
    )
    parser.add_argument("--train_root", type=str, default="data_20260327")
    parser.add_argument("--test_root", type=str, default="rail_mvtec_gt_test")
    parser.add_argument("--view_id", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--precision", type=str, default="fp32", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--depth_norm", type=str, default="zscore", choices=["zscore", "minmax", "log"])
    parser.add_argument("--patch_size", type=int, default=900)
    parser.add_argument("--patch_stride", type=int, default=850)
    parser.add_argument("--use_patch", action="store_true", default=True)
    parser.add_argument("--no_patch", action="store_false", dest="use_patch")
    parser.add_argument("--results_root", type=str, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--cfca_repair_ckpt", type=str, default=DEFAULT_CFCA_REPAIR_CKPT)
    parser.add_argument("--cfca_repair_depth_peft", type=str, default=DEFAULT_CFCA_REPAIR_PEFT)
    parser.add_argument("--chained_ckpt", type=str, default=DEFAULT_CHAINED_CKPT)
    return parser


def make_scheme_specs(args) -> list[dict]:
    return [
        {
            "scheme": "cf_ca_repair_final",
            "ckpt": str(args.cfca_repair_ckpt),
            "depth_peft_ckpt": str(args.cfca_repair_depth_peft),
            "expected_branch_auroc": {"rgb": 0.6750, "depth": 0.8000, "fusion_sum": 0.7750},
        },
        {
            "scheme": "peft_full_then_cf_ca_final",
            "ckpt": str(args.chained_ckpt),
            "depth_peft_ckpt": "",
            "expected_branch_auroc": {"rgb": 0.6875, "depth": 0.8375, "fusion_sum": 0.6500},
        },
    ]


def score_from_map(amap: np.ndarray) -> float:
    return float(np.max(amap))


def zscore_map(amap: np.ndarray) -> np.ndarray:
    amap = np.asarray(amap, dtype=np.float32)
    mean = float(amap.mean())
    std = float(amap.std())
    if not np.isfinite(std) or std <= 1e-8:
        return np.zeros_like(amap, dtype=np.float32)
    return (amap - mean) / std


def ensure_exists(path_str: str, label: str) -> Path:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def load_models_for_scheme(spec: dict, device: torch.device):
    ckpt_path = ensure_exists(spec["ckpt"], f"{spec['scheme']} checkpoint")
    ckpt = safe_load_ckpt(str(ckpt_path), device)

    teacher_rgb = ResNet50Encoder(pretrained=True).to(device).eval()
    teacher_depth = ResNet50Encoder(pretrained=True).to(device).eval()
    for param in teacher_rgb.parameters():
        param.requires_grad_(False)
    for param in teacher_depth.parameters():
        param.requires_grad_(False)

    student_rgb = ResNet50DualModalDecoder(pretrained=False, module_ablation="full").to(device).eval()
    student_depth = ResNet50DualModalDecoder(pretrained=False, module_ablation="full").to(device).eval()
    student_rgb.load_state_dict(strip_prefix(ckpt["student_rgb"]))
    student_depth.load_state_dict(strip_prefix(ckpt["student_depth"]))

    peft_source = ""
    if spec.get("depth_peft_ckpt"):
        peft_path = ensure_exists(spec["depth_peft_ckpt"], f"{spec['scheme']} depth PEFT")
        peft, _ = load_depth_peft(str(peft_path), device)
        teacher_depth = DepthEncoderWithPEFT(teacher_depth, peft).to(device).eval()
        peft_source = str(peft_path)
    else:
        embedded_peft, _ = load_depth_peft_from_joint_ckpt(ckpt, device)
        if embedded_peft is not None:
            teacher_depth = DepthEncoderWithPEFT(teacher_depth, embedded_peft).to(device).eval()
            peft_source = f"{ckpt_path}::embedded"

    return ckpt_path, peft_source, teacher_rgb, teacher_depth, student_rgb, student_depth


@torch.no_grad()
def collect_scheme_diagnostics(spec: dict, args, out_dir: Path) -> dict:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    amp_dtype = resolve_amp_dtype(args.precision, device)
    amp_ctx_factory = amp_context_factory(device, amp_dtype)

    dataset = RailDualModalDataset(
        train_root=args.train_root,
        test_root=args.test_root,
        view_id=args.view_id,
        split="test",
        img_size=args.img_size,
        depth_norm=args.depth_norm,
        use_patch=args.use_patch,
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        preload=False,
        preload_workers=0,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    ckpt_path, peft_source, teacher_rgb, teacher_depth, student_rgb, student_depth = load_models_for_scheme(spec, device)

    per_image_rows: list[dict] = []
    frame_labels: dict[str, int] = {}
    frame_maps: dict[str, list[dict]] = {}
    rgb_global_min = math.inf
    rgb_global_max = -math.inf
    depth_global_min = math.inf
    depth_global_max = -math.inf

    for batch in loader:
        rgb = batch["intensity"].to(device, non_blocking=True)
        depth = batch["depth"].to(device, non_blocking=True)
        labels = batch["label"].cpu().numpy()
        frame_ids = list(batch["frame_id"])
        patch_indices = batch["patch_idx"].cpu().numpy()

        with (amp_ctx_factory() if amp_dtype is not None else nullcontext()):
            feat_t_rgb = teacher_rgb(rgb)
            feat_t_depth = teacher_depth(depth)
            _, _, feat_s_rgb, _ = student_rgb(feat_t_rgb, feat_t_depth)
            _, _, feat_s_depth, _ = student_depth(feat_t_depth, feat_t_rgb)

        amap_rgb, _ = cal_anomaly_map(feat_s_rgb, feat_t_rgb, out_size=(256, 256), amap_mode="mul")
        amap_depth, _ = cal_anomaly_map(feat_s_depth, feat_t_depth, out_size=(256, 256), amap_mode="mul")
        amap_rgb = np.asarray(amap_rgb, dtype=np.float32)
        amap_depth = np.asarray(amap_depth, dtype=np.float32)

        rgb_global_min = min(rgb_global_min, float(np.min(amap_rgb)))
        rgb_global_max = max(rgb_global_max, float(np.max(amap_rgb)))
        depth_global_min = min(depth_global_min, float(np.min(amap_depth)))
        depth_global_max = max(depth_global_max, float(np.max(amap_depth)))

        for idx, frame_id in enumerate(frame_ids):
            label = int(labels[idx])
            patch_idx = int(patch_indices[idx])
            rgb_map = amap_rgb[idx]
            depth_map = amap_depth[idx]
            rgb_max = float(np.max(rgb_map))
            depth_max = float(np.max(depth_map))
            ratio = depth_max / max(rgb_max, 1e-8)
            per_image_rows.append(
                {
                    "scheme": spec["scheme"],
                    "frame_id": frame_id,
                    "label": label,
                    "patch_idx": patch_idx,
                    "rgb_max": f"{rgb_max:.8f}",
                    "depth_max": f"{depth_max:.8f}",
                    "depth_rgb_ratio": f"{ratio:.8f}",
                }
            )
            frame_labels.setdefault(frame_id, label)
            frame_maps.setdefault(frame_id, []).append(
                {
                    "patch_idx": patch_idx,
                    "rgb": rgb_map,
                    "depth": depth_map,
                }
            )

    scale_per_image_path = out_dir / "scale_per_image.csv"
    write_csv(
        scale_per_image_path,
        per_image_rows,
        ["scheme", "frame_id", "label", "patch_idx", "rgb_max", "depth_max", "depth_rgb_ratio"],
    )

    ordered_frame_ids = sorted(frame_labels.keys())
    labels = np.array([frame_labels[fid] for fid in ordered_frame_ids], dtype=np.int64)

    fusion_rows: list[dict] = []
    fusion_summary_rows: list[dict] = []
    sum_scores = []
    zscore_scores = []
    max_norm_scores = []

    rgb_denom = max(rgb_global_max - rgb_global_min, 1e-8)
    depth_denom = max(depth_global_max - depth_global_min, 1e-8)

    for frame_id in ordered_frame_ids:
        patches = sorted(frame_maps[frame_id], key=lambda item: item["patch_idx"])
        sum_patch_scores = []
        zscore_patch_scores = []
        max_norm_patch_scores = []

        for item in patches:
            rgb_map = item["rgb"]
            depth_map = item["depth"]
            fusion_sum = rgb_map + depth_map
            fusion_zscore = zscore_map(rgb_map) + zscore_map(depth_map)
            rgb_norm = (rgb_map - rgb_global_min) / rgb_denom
            depth_norm = (depth_map - depth_global_min) / depth_denom
            fusion_max_norm = np.maximum(rgb_norm, depth_norm)

            sum_patch_scores.append(score_from_map(fusion_sum))
            zscore_patch_scores.append(score_from_map(fusion_zscore))
            max_norm_patch_scores.append(score_from_map(fusion_max_norm))

        image_sum = float(max(sum_patch_scores))
        image_zscore = float(max(zscore_patch_scores))
        image_max_norm = float(max(max_norm_patch_scores))
        sum_scores.append(image_sum)
        zscore_scores.append(image_zscore)
        max_norm_scores.append(image_max_norm)
        fusion_rows.append(
            {
                "scheme": spec["scheme"],
                "frame_id": frame_id,
                "label": int(frame_labels[frame_id]),
                "fusion_sum_score": f"{image_sum:.8f}",
                "fusion_zscore_sum_score": f"{image_zscore:.8f}",
                "fusion_max_norm_score": f"{image_max_norm:.8f}",
            }
        )

    fusion_scores_path = out_dir / f"fusion_scores_{spec['scheme']}.csv"
    write_csv(
        fusion_scores_path,
        fusion_rows,
        [
            "scheme",
            "frame_id",
            "label",
            "fusion_sum_score",
            "fusion_zscore_sum_score",
            "fusion_max_norm_score",
        ],
    )

    auroc_sum = float(roc_auc_score(labels, np.array(sum_scores, dtype=np.float64)))
    auroc_zscore = float(roc_auc_score(labels, np.array(zscore_scores, dtype=np.float64)))
    auroc_max_norm = float(roc_auc_score(labels, np.array(max_norm_scores, dtype=np.float64)))
    for rule_name, value in [
        ("sum", auroc_sum),
        ("zscore_sum", auroc_zscore),
        ("max_norm", auroc_max_norm),
    ]:
        fusion_summary_rows.append(
            {
                "scheme": spec["scheme"],
                "fusion_rule": rule_name,
                "fusion_auroc": f"{value:.8f}",
                "delta_vs_sum_same_scheme": f"{(value - auroc_sum):.8f}",
                "num_images": str(int(labels.size)),
                "num_abnormal": str(int((labels == 1).sum())),
                "num_normal": str(int((labels == 0).sum())),
            }
        )

    scale_values = np.array([float(row["depth_rgb_ratio"]) for row in per_image_rows], dtype=np.float64)
    rgb_max_values = np.array([float(row["rgb_max"]) for row in per_image_rows], dtype=np.float64)
    depth_max_values = np.array([float(row["depth_max"]) for row in per_image_rows], dtype=np.float64)
    scale_summary = {
        "scheme": spec["scheme"],
        "rgb_max_mean": f"{float(rgb_max_values.mean()):.8f}",
        "rgb_max_median": f"{float(np.median(rgb_max_values)):.8f}",
        "depth_max_mean": f"{float(depth_max_values.mean()):.8f}",
        "depth_max_median": f"{float(np.median(depth_max_values)):.8f}",
        "ratio_mean": f"{float(scale_values.mean()):.8f}",
        "ratio_median": f"{float(np.median(scale_values)):.8f}",
        "ratio_p25": f"{float(np.percentile(scale_values, 25.0)):.8f}",
        "ratio_p75": f"{float(np.percentile(scale_values, 75.0)):.8f}",
    }

    return {
        "scheme": spec["scheme"],
        "ckpt": str(ckpt_path),
        "depth_peft_source": peft_source,
        "expected_branch_auroc": spec["expected_branch_auroc"],
        "frame_ids": ordered_frame_ids,
        "num_images": int(labels.size),
        "num_abnormal": int((labels == 1).sum()),
        "num_normal": int((labels == 0).sum()),
        "scale_summary": scale_summary,
        "fusion_summary_rows": fusion_summary_rows,
        "scale_per_image_path": str(scale_per_image_path),
        "fusion_scores_path": str(fusion_scores_path),
        "observed_fusion_auroc": {
            "sum": auroc_sum,
            "zscore_sum": auroc_zscore,
            "max_norm": auroc_max_norm,
        },
        "scale_rows": per_image_rows,
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = build_parser().parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.results_root) / f"cam4_fusion_mechanism_validation_{timestamp}"
    out_root.mkdir(parents=True, exist_ok=True)

    summary_payload = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "note": "zscore_sum uses per-image map-level z-score before fusion; not the paper main metric.",
        "args": {
            "train_root": str(args.train_root),
            "test_root": str(args.test_root),
            "view_id": int(args.view_id),
            "img_size": int(args.img_size),
            "depth_norm": str(args.depth_norm),
            "patch_size": int(args.patch_size),
            "patch_stride": int(args.patch_stride),
            "precision": str(args.precision),
        },
        "schemes": [],
    }

    scale_summary_rows = []
    fusion_summary_rows = []
    missing = []
    for spec in make_scheme_specs(args):
        scheme_dir = out_root / spec["scheme"]
        scheme_dir.mkdir(parents=True, exist_ok=True)
        try:
            payload = collect_scheme_diagnostics(spec, args, scheme_dir)
        except FileNotFoundError as exc:
            missing.append(str(exc))
            summary_payload["schemes"].append(
                {
                    "scheme": spec["scheme"],
                    "status": "missing_checkpoint",
                    "error": str(exc),
                    "requested_ckpt": spec["ckpt"],
                    "requested_depth_peft_ckpt": spec.get("depth_peft_ckpt", ""),
                }
            )
            continue

        summary_payload["schemes"].append(
            {
                "scheme": payload["scheme"],
                "status": "ok",
                "ckpt": payload["ckpt"],
                "depth_peft_source": payload["depth_peft_source"],
                "expected_branch_auroc": payload["expected_branch_auroc"],
                "observed_fusion_auroc": payload["observed_fusion_auroc"],
                "num_images": payload["num_images"],
                "num_abnormal": payload["num_abnormal"],
                "num_normal": payload["num_normal"],
                "scale_summary": payload["scale_summary"],
                "scale_per_image_csv": payload["scale_per_image_path"],
                "fusion_scores_csv": payload["fusion_scores_path"],
            }
        )
        scale_summary_rows.append(payload["scale_summary"])
        fusion_summary_rows.extend(payload["fusion_summary_rows"])

    if scale_summary_rows:
        write_csv(
            out_root / "scale_summary.csv",
            scale_summary_rows,
            [
                "scheme",
                "rgb_max_mean",
                "rgb_max_median",
                "depth_max_mean",
                "depth_max_median",
                "ratio_mean",
                "ratio_median",
                "ratio_p25",
                "ratio_p75",
            ],
        )
    if fusion_summary_rows:
        write_csv(
            out_root / "fusion_summary.csv",
            fusion_summary_rows,
            [
                "scheme",
                "fusion_rule",
                "fusion_auroc",
                "delta_vs_sum_same_scheme",
                "num_images",
                "num_abnormal",
                "num_normal",
            ],
        )

    summary_payload["status"] = "partial_missing" if missing else "ok"
    summary_payload["missing"] = missing
    summary_path = out_root / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)

    print(f"Output dir: {out_root}")
    if missing:
        print("Missing checkpoints:")
        for item in missing:
            print(f"- {item}")
    else:
        print("Validation completed successfully.")


if __name__ == "__main__":
    main()
