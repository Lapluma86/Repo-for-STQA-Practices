#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compatibility wrapper for legacy PatchCore Cam4 entrypoint."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.random_projection import SparseRandomProjection

try:
    import tifffile
except ImportError:
    tifffile = None

# Use same feature extractor as TRD
from models.trd.encoder import ResNet50Encoder


def extract_features(encoder: torch.nn.Module, loader: DataLoader,
                     device: torch.device, layers: tuple = (1, 2),
                     max_samples: int = 200) -> dict[int, list[np.ndarray]]:
    """Extract multi-layer features from normal samples."""
    encoder.eval()
    features: dict[int, list[np.ndarray]] = {}
    count = 0
    with torch.no_grad():
        for rgb, _, _, _ in loader:
            if count >= max_samples:
                break
            rgb = rgb.to(device)
            feats = encoder(rgb)  # list of tensors
            for layer_idx in layers:
                f = feats[layer_idx].cpu().numpy()
                if layer_idx not in features:
                    features[layer_idx] = []
                features[layer_idx].append(f)
            count += 1
    return features


def build_memory_bank(features: dict[int, list[np.ndarray]],
                      coreset_ratio: float = 0.10,
                      random_seed: int = 42) -> dict[int, np.ndarray]:
    """Aggregate features into memory bank with coreset subsampling."""
    memory = {}
    rng = np.random.RandomState(random_seed)
    for layer_idx, feat_list in features.items():
        all_feats = np.concatenate(feat_list, axis=0)  # [N, C, H, W]
        N, C, H, W = all_feats.shape
        # Reshape: [N*H*W, C]
        patch_feats = all_feats.transpose(0, 2, 3, 1).reshape(-1, C)
        # Coreset via random sampling
        n_coreset = max(1, int(len(patch_feats) * coreset_ratio))
        indices = rng.choice(len(patch_feats), n_coreset, replace=False)
        memory[layer_idx] = patch_feats[indices]
    return memory


def score_patchcore(encoder: torch.nn.Module, loader: DataLoader,
                    memory: dict[int, np.ndarray],
                    device: torch.device, layers: tuple = (1, 2),
                    k: int = 5, chunk_size: int = 256) -> tuple[list[float], list[np.ndarray], list[str]]:
    """Compute PatchCore anomaly scores with chunked distance computation."""
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
                f = feats[layer_idx].cpu().numpy()  # [B, C, H, W]
                B, C, H, W = f.shape
                mem = memory[layer_idx]  # [M, C]
                f_flat = f.transpose(0, 2, 3, 1).reshape(B, H * W, C)

                scores = np.zeros((B, H * W))
                for b in range(B):
                    n_patches = H * W
                    # Chunked: avoid (n_patches, M, C) memory explosion
                    for start in range(0, n_patches, chunk_size):
                        end = min(start + chunk_size, n_patches)
                        chunk = f_flat[b, start:end]  # [chunk, C]
                        # dists: [chunk, M] via linalg.norm along last axis
                        diff = chunk[:, None, :] - mem[None, :, :]  # [chunk, M, C]
                        dists = np.sqrt(np.sum(diff * diff, axis=-1))  # [chunk, M]
                        topk = np.sort(dists, axis=-1)[:, :k]
                        scores[b, start:end] = np.mean(topk, axis=-1)

                score_map = scores.reshape(B, H, W)
                batch_maps.append(score_map)

            combined = np.mean(batch_maps, axis=0)
            for b in range(B):
                smap = combined[b]
                image_scores.append(float(np.max(smap)))
                pixel_maps.append(smap)
                frame_ids.append(fids[b])

    return image_scores, pixel_maps, frame_ids


class RailRGBDataset(torch.utils.data.Dataset):
    """RGB-only dataset for PatchCore/PaDiM evaluation."""

    def __init__(self, root: Path, cam: str, split: str = "test"):
        self.root = Path(root)
        self.cam_dir = self.root / "rail_mvtec" / cam / split
        self.samples: list[tuple[Path, int, str]] = []  # (path, label, frame_id)
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])

        for label_name, label_val in [("good", 0), ("broken", 1)]:
            label_dir = self.cam_dir / label_name
            if not label_dir.exists():
                continue
            for img_path in sorted(label_dir.glob("*.jpg")):
                self.samples.append((img_path, label_val, img_path.stem))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label, fid = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img_tensor = self.transform(img)
        return img_tensor, label, path.name, fid
PROJ_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="PatchCore baseline compatibility wrapper")
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
    parser.add_argument("--projection_dim", "--patchcore_projection_dim", dest="projection_dim", type=int, default=128)
    parser.add_argument("--coreset_ratio", "--patchcore_coreset_ratio", dest="coreset_ratio", type=float, default=0.01)
    parser.add_argument("--coreset_size", "--patchcore_coreset_size", dest="coreset_size", type=int, default=0)
    parser.add_argument("--k", "--patchcore_k", dest="k", type=int, default=5)
    parser.add_argument("--query_chunk_size", "--patchcore_query_chunk_size", dest="query_chunk_size", type=int, default=2048)
    parser.add_argument("--layers", "--patchcore_layers", dest="layers", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--target_map_size", "--patchcore_target_map_size", dest="target_map_size", type=int, default=32)
    args = parser.parse_args()

    view_id = args.view_id
    if view_id is None:
        cam = str(args.cam).strip().lower()
        view_id = int(cam[3:]) if cam.startswith("cam") else int(cam)

    cmd = [
        sys.executable,
        str(PROJ_ROOT / "scripts" / "baselines" / "run_rgb_baseline.py"),
        "--method",
        "patchcore",
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
        "--patchcore_projection_dim",
        str(args.projection_dim),
        "--patchcore_coreset_ratio",
        str(args.coreset_ratio),
        "--patchcore_coreset_size",
        str(args.coreset_size),
        "--patchcore_k",
        str(args.k),
        "--patchcore_query_chunk_size",
        str(args.query_chunk_size),
        "--patchcore_target_map_size",
        str(args.target_map_size),
        "--patchcore_layers",
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
