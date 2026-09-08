#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compatibility wrapper for legacy PaDiM Cam4 entrypoint."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from scripts.baselines.run_patchcore_cam4 import (
    RailRGBDataset, extract_features,
)


def estimate_gaussians(features: dict[int, list[np.ndarray]],
                       layers: tuple = (1, 2)) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Estimate per-patch multivariate Gaussian (mean, inv_cov)."""
    params = {}
    for layer_idx in layers:
        feat_list = features[layer_idx]
        # Stack: [N, C, H, W]
        all_feats = np.concatenate(feat_list, axis=0)  # [N, C, H, W]
        N, C, H, W = all_feats.shape
        # [H*W, N, C]
        patches = all_feats.transpose(0, 2, 3, 1).reshape(N, H * W, C).transpose(1, 0, 2)
        means = np.zeros((H * W, C))
        inv_covs = np.zeros((H * W, C, C))
        eps = 1e-4
        for p in range(H * W):
            patch_data = patches[p]  # [N, C]
            mu = patch_data.mean(axis=0)
            cov = np.cov(patch_data, rowvar=False)
            cov += eps * np.eye(C)
            means[p] = mu
            inv_covs[p] = np.linalg.inv(cov)
        params[layer_idx] = (means, inv_covs)
    return params


def score_padim(encoder: torch.nn.Module, loader: DataLoader,
                gaussian_params: dict[int, tuple[np.ndarray, np.ndarray]],
                device: torch.device, layers: tuple = (1, 2),
                ) -> tuple[list[float], list[np.ndarray], list[str]]:
    """Compute PaDiM anomaly scores using vectorized Mahalanobis distance."""
    encoder.eval()
    image_scores: list[float] = []
    pixel_maps: list[np.ndarray] = []
    frame_ids: list[str] = []

    with torch.no_grad():
        for rgb, _, _, fids in loader:
            rgb = rgb.to(device)
            feats = encoder(rgb)
            batch_maps = []
            for layer_idx in layers:
                (means, inv_covs) = gaussian_params[layer_idx]  # means:[P,C], inv_covs:[P,C,C]
                P, C = means.shape
                f = feats[layer_idx].cpu().numpy()  # [B, C, H, W]
                B, Cf, H, W_ = f.shape
                assert C == Cf and P == H * W_, f"Shape mismatch: means {P}x{C}, feat {H}x{W_}x{Cf}"

                f_patches = f.transpose(0, 2, 3, 1).reshape(B, H * W_, C)  # [B, P, C]
                diff = f_patches - means[None, :, :]  # [B, P, C]
                # Vectorized Mahalanobis: for each (b,p), diff[b,p] @ inv_covs[p] @ diff[b,p]
                # = sum_c(diff[b,p,c] * sum_c'(inv_covs[p,c,c'] * diff[b,p,c']))
                # = einsum('bpc,pcd,bpd->bp', diff, inv_covs, diff)
                scores = np.sqrt(np.maximum(
                    np.einsum('bpc,pcd,bpd->bp', diff, inv_covs, diff), 0.0
                ))  # [B, P]
                score_map = scores.reshape(B, H, W_)
                batch_maps.append(score_map)

            combined = np.max(batch_maps, axis=0)  # [B, H, W]
            for b in range(B):
                image_scores.append(float(np.max(combined[b])))
                pixel_maps.append(combined[b])
                frame_ids.append(fids[b])

    return image_scores, pixel_maps, frame_ids
PROJ_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="PaDiM baseline compatibility wrapper")
    parser.add_argument("--train_root", type=str, default="/data1/Leaddo_data/20260327-resize512")
    parser.add_argument("--test_root", "--eval_root", dest="test_root", type=str, default="/home/root123/LF/WYM/SteelRailWay/rail_mvtec_gt_test")
    parser.add_argument("--cam", type=str, default="cam4")
    parser.add_argument("--view_id", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--patch_size", type=int, default=900)
    parser.add_argument("--patch_stride", type=int, default=850)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--preload", action="store_true")
    parser.add_argument("--preload_workers", type=int, default=4)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--scores_csv", type=str, default="")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--precision", type=str, default="bf16")
    parser.add_argument("--train_sample_num", type=int, default=1500)
    parser.add_argument("--train_val_test_split", type=float, nargs=3, default=[0.9, 0.1, 0.0])
    parser.add_argument("--sampling_mode", type=str, default="uniform_time")
    parser.add_argument("--layers", "--padim_layers", dest="layers", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--target_map_size", "--padim_target_map_size", dest="target_map_size", type=int, default=16)
    args = parser.parse_args()

    view_id = args.view_id
    if view_id is None:
        cam = str(args.cam).strip().lower()
        view_id = int(cam[3:]) if cam.startswith("cam") else int(cam)

    cmd = [
        sys.executable,
        str(PROJ_ROOT / "scripts" / "baselines" / "run_rgb_baseline.py"),
        "--method",
        "padim",
        "--train_root",
        args.train_root,
        "--test_root",
        args.test_root,
        "--view_id",
        str(view_id),
        "--img_size",
        str(args.img_size),
        "--train_sample_num",
        str(args.train_sample_num),
        "--train_val_test_split",
        *[str(x) for x in args.train_val_test_split],
        "--sampling_mode",
        args.sampling_mode,
        "--patch_size",
        str(args.patch_size),
        "--patch_stride",
        str(args.patch_stride),
        "--device",
        args.device,
        "--precision",
        args.precision,
        "--batch_size",
        str(args.batch_size),
        "--eval_batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
        "--preload_workers",
        str(args.preload_workers),
        "--padim_target_map_size",
        str(args.target_map_size),
        "--padim_layers",
        *[str(x) for x in args.layers],
    ]
    if args.preload:
        cmd.append("--preload")
    if args.smoke:
        cmd.append("--smoke")
    if args.output:
        cmd.extend(["--result_json", args.output])
    if args.scores_csv:
        cmd.extend(["--scores_csv", args.scores_csv])
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
