#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unified RGB-only baseline launcher for Cam4-oriented comparison experiments."""

from __future__ import annotations

import argparse
import math
import sys
from contextlib import nullcontext
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from sklearn.random_projection import SparseRandomProjection

from eval.eval_utils import cal_anomaly_map
from models.trd.decoder import ResNet50Decoder
from models.trd.encoder import ResNet50Encoder
from scripts.baselines.common_protocol import (
    ProtocolConfig,
    build_loader,
    build_protocol_datasets,
    compute_safe_auroc,
    dataset_summary,
    default_result_paths,
    make_method_result,
    precision_to_dtype,
    reduce_patch_predictions,
    seed_everything,
    write_result_json,
    write_scores_csv,
    write_summary_files,
)
from utils.losses import loss_distil, loss_l2


METHODS = ("padim", "patchcore", "fastflow", "draem", "stfpm", "rd4ad_i")
METHOD_BATCH_DEFAULTS = {
    "padim": {"batch_size": 8, "eval_batch_size": 8},
    "patchcore": {"batch_size": 8, "eval_batch_size": 8},
    "fastflow": {"batch_size": 32, "eval_batch_size": 8},
    "draem": {"batch_size": 16, "eval_batch_size": 8},
    "stfpm": {"batch_size": 32, "eval_batch_size": 8},
    "rd4ad_i": {"batch_size": 32, "eval_batch_size": 8},
}


def device_autocast(device: torch.device, dtype: torch.dtype | None):
    if dtype is None or device.type != "cuda":
        return nullcontext()
    try:
        from torch.amp import autocast

        return autocast(device_type=device.type, dtype=dtype)
    except Exception:
        try:
            from torch.cuda.amp import autocast

            return autocast(dtype=dtype)
        except Exception:
            return nullcontext()


def freeze_module(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        p.requires_grad_(False)
    return module.eval()


def parse_layers(values: list[int] | tuple[int, ...]) -> tuple[int, ...]:
    layers = tuple(int(v) for v in values)
    if not layers:
        raise ValueError("At least one layer index is required")
    if min(layers) < 0 or max(layers) > 2:
        raise ValueError(f"Encoder emits 3 feature maps only; got layers={layers}")
    return layers


def ensure_method_defaults(args, method: str) -> None:
    defaults = METHOD_BATCH_DEFAULTS[method]
    if args.batch_size <= 0:
        args.batch_size = defaults["batch_size"]
    if args.eval_batch_size <= 0:
        args.eval_batch_size = defaults["eval_batch_size"]


def resolve_method_batch_sizes(args, method: str) -> tuple[int, int]:
    defaults = METHOD_BATCH_DEFAULTS[method]
    batch_size = args.batch_size if args.batch_size > 0 else defaults["batch_size"]
    eval_batch_size = args.eval_batch_size if args.eval_batch_size > 0 else defaults["eval_batch_size"]
    return int(batch_size), int(eval_batch_size)


def extract_embeddings(
    encoder: nn.Module,
    image: torch.Tensor,
    *,
    layers: tuple[int, ...],
    target_map_size: int,
) -> tuple[torch.Tensor, int, int, int]:
    feats = encoder(image)
    selected = []
    for layer_idx in layers:
        feat = feats[layer_idx]
        if target_map_size > 0 and feat.shape[-1] != target_map_size:
            feat = F.adaptive_avg_pool2d(feat, (target_map_size, target_map_size))
        selected.append(feat)
    embedding = torch.cat(selected, dim=1)
    embedding = F.normalize(embedding, dim=1)
    batch, channels, height, width = embedding.shape
    embedding = embedding.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
    return embedding, channels, height, width


def batched_knn_distances(index: NearestNeighbors, queries: np.ndarray, chunk_size: int) -> np.ndarray:
    chunks = []
    for start in range(0, len(queries), chunk_size):
        distances, _ = index.kneighbors(queries[start:start + chunk_size], return_distance=True)
        chunks.append(distances.astype(np.float32, copy=False))
    return np.concatenate(chunks, axis=0)


def collect_memory_features(
    encoder: nn.Module,
    loader,
    *,
    device: torch.device,
    layers: tuple[int, ...],
    target_map_size: int,
    amp_dtype: torch.dtype | None,
) -> tuple[np.ndarray, int]:
    encoder.eval()
    chunks: list[np.ndarray] = []
    channels = 0
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            with device_autocast(device, amp_dtype):
                embedding, channels, _, _ = extract_embeddings(
                    encoder,
                    image,
                    layers=layers,
                    target_map_size=target_map_size,
                )
            chunks.append(
                embedding.reshape(-1, embedding.shape[-1]).detach().float().cpu().numpy().astype(np.float32)
            )
    if not chunks:
        raise RuntimeError("No memory-bank features were extracted")
    return np.concatenate(chunks, axis=0), int(channels)


def build_memory_bank(
    features: np.ndarray,
    *,
    coreset_ratio: float,
    coreset_size: int,
    projection_dim: int,
    random_seed: int,
) -> tuple[np.ndarray, SparseRandomProjection | None, np.ndarray]:
    projector: SparseRandomProjection | None = None
    projected = features.astype(np.float32, copy=False)
    if projection_dim > 0 and projection_dim < projected.shape[1]:
        projector = SparseRandomProjection(
            n_components=projection_dim,
            dense_output=True,
            random_state=random_seed,
        )
        projected = projector.fit_transform(projected).astype(np.float32, copy=False)

    total = int(projected.shape[0])
    if coreset_size > 0:
        keep = min(total, int(coreset_size))
    else:
        keep = max(1, int(round(total * float(coreset_ratio))))
        keep = min(total, keep)
    rng = np.random.RandomState(random_seed)
    if keep == total:
        indices = np.arange(total, dtype=np.int64)
    else:
        indices = np.sort(rng.choice(total, size=keep, replace=False))
    return projected[indices], projector, indices


def evaluate_patchcore(config: ProtocolConfig, datasets: dict, args) -> dict:
    device = torch.device(config.device)
    amp_dtype = precision_to_dtype(config.precision)
    encoder = freeze_module(ResNet50Encoder(pretrained=True).to(device))
    train_loader = build_loader(datasets["train"], batch_size=config.eval_batch_size, num_workers=config.num_workers, device=device)
    test_loader = build_loader(datasets["test"], batch_size=config.eval_batch_size, num_workers=config.num_workers, device=device)
    layers = parse_layers(args.patchcore_layers)

    memory_full, embedding_dim = collect_memory_features(
        encoder,
        train_loader,
        device=device,
        layers=layers,
        target_map_size=args.patchcore_target_map_size,
        amp_dtype=amp_dtype,
    )
    memory_bank, projector, coreset_indices = build_memory_bank(
        memory_full,
        coreset_ratio=args.patchcore_coreset_ratio,
        coreset_size=args.patchcore_coreset_size,
        projection_dim=args.patchcore_projection_dim,
        random_seed=config.seed,
    )
    index = NearestNeighbors(
        n_neighbors=int(args.patchcore_k),
        algorithm="auto",
        metric="euclidean",
        n_jobs=-1,
    )
    index.fit(memory_bank)

    frame_scores: dict[str, list[float]] = {}
    frame_labels: dict[str, int] = {}
    patch_score_chunks: list[np.ndarray] = []
    patch_gt_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for batch in test_loader:
            image = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].cpu().numpy().astype(np.int64)
            gt = batch["gt"].cpu().numpy().astype(np.float32)
            frame_ids = list(batch["frame_id"])
            with device_autocast(device, amp_dtype):
                embedding, _, h, w = extract_embeddings(
                    encoder,
                    image,
                    layers=layers,
                    target_map_size=args.patchcore_target_map_size,
                )
            queries = embedding.reshape(-1, embedding.shape[-1]).detach().float().cpu().numpy().astype(np.float32)
            if projector is not None:
                queries = projector.transform(queries).astype(np.float32, copy=False)
            distances = batched_knn_distances(index, queries, args.patchcore_query_chunk_size)
            patch_maps = distances.mean(axis=1).reshape(len(frame_ids), h, w)
            patch_maps = torch.from_numpy(patch_maps).unsqueeze(1)
            patch_maps = F.interpolate(
                patch_maps,
                size=(config.img_size, config.img_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1).numpy()
            patch_score_chunks.append(patch_maps.reshape(len(frame_ids), -1))
            patch_gt_chunks.append(gt.reshape(len(frame_ids), -1))
            for idx, frame_id in enumerate(frame_ids):
                frame_scores.setdefault(frame_id, []).append(float(patch_maps[idx].max()))
                frame_labels[frame_id] = int(labels[idx])

    image_scores, image_labels, frame_ids, pixel_scores, pixel_labels = reduce_patch_predictions(
        frame_scores, frame_labels, patch_score_chunks, patch_gt_chunks
    )
    image_auroc = compute_safe_auroc(image_labels, image_scores)
    pixel_auroc = compute_safe_auroc(pixel_labels.astype(np.int64), pixel_scores)
    result_json, scores_csv = default_result_paths(config)
    write_scores_csv(scores_csv, frame_ids, image_labels, image_scores)
    payload = make_method_result(
        config=config,
        image_auroc=image_auroc,
        pixel_auroc=pixel_auroc,
        train_summary=dataset_summary(datasets["train"]),
        val_summary=dataset_summary(datasets["val"]),
        test_summary=dataset_summary(datasets["test"]),
        scores_csv=scores_csv,
        extra={
            "layers": list(layers),
            "target_map_size": int(args.patchcore_target_map_size),
            "embedding_dim_before_projection": int(embedding_dim),
            "embedding_dim_after_projection": int(memory_bank.shape[1]),
            "projection_dim_arg": int(args.patchcore_projection_dim),
            "coreset_ratio": float(args.patchcore_coreset_ratio),
            "coreset_size_arg": int(args.patchcore_coreset_size),
            "memory_vectors_full": int(memory_full.shape[0]),
            "memory_vectors_coreset": int(memory_bank.shape[0]),
            "memory_coreset_indices_count": int(len(coreset_indices)),
            "k_nn": int(args.patchcore_k),
            "query_chunk_size": int(args.patchcore_query_chunk_size),
        },
    )
    write_result_json(result_json, payload)
    payload["result_json"] = str(result_json)
    return payload


def collect_layer_features(
    encoder: nn.Module,
    loader,
    *,
    device: torch.device,
    layers: tuple[int, ...],
    target_map_size: int,
    amp_dtype: torch.dtype | None,
) -> dict[int, np.ndarray]:
    encoder.eval()
    collected: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            with device_autocast(device, amp_dtype):
                feats = encoder(image)
            for layer_idx in layers:
                feat = feats[layer_idx]
                if target_map_size > 0 and feat.shape[-1] != target_map_size:
                    feat = F.adaptive_avg_pool2d(feat, (target_map_size, target_map_size))
                collected[layer_idx].append(feat.detach().float().cpu().numpy().astype(np.float32))
    return {layer: np.concatenate(chunks, axis=0) for layer, chunks in collected.items()}


def estimate_gaussians(
    features: dict[int, np.ndarray],
    *,
    diagonal_only: bool = False,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    params: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    eps = 1e-4
    for layer_idx, feat in features.items():
        n, c, h, w = feat.shape
        patches = feat.transpose(0, 2, 3, 1).reshape(n, h * w, c).transpose(1, 0, 2)
        means = patches.mean(axis=1).astype(np.float32)
        if diagonal_only:
            inv_covs = np.zeros((h * w, c), dtype=np.float32)
        else:
            inv_covs = np.zeros((h * w, c, c), dtype=np.float32)
        for patch_idx in range(h * w):
            patch_data = patches[patch_idx]
            if diagonal_only:
                var = patch_data.var(axis=0).astype(np.float32) + eps
                inv_covs[patch_idx] = (1.0 / var).astype(np.float32)
            else:
                cov = np.cov(patch_data, rowvar=False).astype(np.float32)
                cov += eps * np.eye(c, dtype=np.float32)
                inv_covs[patch_idx] = np.linalg.inv(cov).astype(np.float32)
        params[layer_idx] = (means, inv_covs)
    return params


def evaluate_padim(config: ProtocolConfig, datasets: dict, args) -> dict:
    device = torch.device(config.device)
    amp_dtype = precision_to_dtype(config.precision)
    encoder = freeze_module(ResNet50Encoder(pretrained=True).to(device))
    train_loader = build_loader(datasets["train"], batch_size=config.eval_batch_size, num_workers=config.num_workers, device=device)
    test_loader = build_loader(datasets["test"], batch_size=config.eval_batch_size, num_workers=config.num_workers, device=device)
    layers = parse_layers(args.padim_layers)
    train_features = collect_layer_features(
        encoder,
        train_loader,
        device=device,
        layers=layers,
        target_map_size=args.padim_target_map_size,
        amp_dtype=amp_dtype,
    )
    params = estimate_gaussians(train_features, diagonal_only=config.smoke)

    frame_scores: dict[str, list[float]] = {}
    frame_labels: dict[str, int] = {}
    patch_score_chunks: list[np.ndarray] = []
    patch_gt_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for batch in test_loader:
            image = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].cpu().numpy().astype(np.int64)
            gt = batch["gt"].cpu().numpy().astype(np.float32)
            frame_ids = list(batch["frame_id"])
            with device_autocast(device, amp_dtype):
                feats = encoder(image)

            batch_maps = []
            for layer_idx in layers:
                feat = feats[layer_idx]
                if args.padim_target_map_size > 0 and feat.shape[-1] != args.padim_target_map_size:
                    feat = F.adaptive_avg_pool2d(feat, (args.padim_target_map_size, args.padim_target_map_size))
                feat_np = feat.detach().float().cpu().numpy().astype(np.float32)
                bsz, channels, h, w = feat_np.shape
                means, inv_covs = params[layer_idx]
                feat_patches = feat_np.transpose(0, 2, 3, 1).reshape(bsz, h * w, channels)
                scores = np.zeros((bsz, h * w), dtype=np.float32)
                diagonal_only = inv_covs.ndim == 2
                for b in range(bsz):
                    for patch_idx in range(h * w):
                        diff = feat_patches[b, patch_idx] - means[patch_idx]
                        if diagonal_only:
                            scores[b, patch_idx] = float(np.sqrt((diff * diff * inv_covs[patch_idx]).sum()))
                        else:
                            scores[b, patch_idx] = float(np.sqrt(diff @ inv_covs[patch_idx] @ diff))
                batch_maps.append(scores.reshape(bsz, h, w))

            combined = np.max(np.stack(batch_maps, axis=0), axis=0)
            combined = torch.from_numpy(combined).unsqueeze(1)
            combined = F.interpolate(
                combined,
                size=(config.img_size, config.img_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1).numpy()
            patch_score_chunks.append(combined.reshape(len(frame_ids), -1))
            patch_gt_chunks.append(gt.reshape(len(frame_ids), -1))
            for idx, frame_id in enumerate(frame_ids):
                frame_scores.setdefault(frame_id, []).append(float(combined[idx].max()))
                frame_labels[frame_id] = int(labels[idx])

    image_scores, image_labels, frame_ids, pixel_scores, pixel_labels = reduce_patch_predictions(
        frame_scores, frame_labels, patch_score_chunks, patch_gt_chunks
    )
    image_auroc = compute_safe_auroc(image_labels, image_scores)
    pixel_auroc = compute_safe_auroc(pixel_labels.astype(np.int64), pixel_scores)
    result_json, scores_csv = default_result_paths(config)
    write_scores_csv(scores_csv, frame_ids, image_labels, image_scores)
    payload = make_method_result(
        config=config,
        image_auroc=image_auroc,
        pixel_auroc=pixel_auroc,
        train_summary=dataset_summary(datasets["train"]),
        val_summary=dataset_summary(datasets["val"]),
        test_summary=dataset_summary(datasets["test"]),
        scores_csv=scores_csv,
        extra={
            "layers": list(layers),
            "target_map_size": int(args.padim_target_map_size),
            "covariance_mode": "diagonal_smoke" if config.smoke else "full",
        },
    )
    write_result_json(result_json, payload)
    payload["result_json"] = str(result_json)
    return payload


def feature_distance_map(fs_list, ft_list, out_size: int) -> torch.Tensor:
    batch_maps = []
    for fs, ft in zip(fs_list, ft_list):
        amap = 1 - F.cosine_similarity(fs.float(), ft.float(), dim=1)
        amap = F.interpolate(amap.unsqueeze(1), size=(out_size, out_size), mode="bilinear", align_corners=True)
        batch_maps.append(amap)
    return torch.stack(batch_maps, dim=0).mean(dim=0).squeeze(1)


def evaluate_teacher_student(
    teacher: nn.Module,
    student: nn.Module,
    loader,
    *,
    device: torch.device,
    img_size: int,
    amp_dtype: torch.dtype | None,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray]:
    teacher.eval()
    student.eval()
    frame_scores: dict[str, list[float]] = {}
    frame_labels: dict[str, int] = {}
    patch_score_chunks: list[np.ndarray] = []
    patch_gt_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].cpu().numpy().astype(np.int64)
            gt = batch["gt"].cpu().numpy().astype(np.float32)
            frame_ids = list(batch["frame_id"])
            with device_autocast(device, amp_dtype):
                feat_t = teacher(image)
                feat_s = student(feat_t)
            amap, _ = cal_anomaly_map(feat_s, feat_t, out_size=(img_size, img_size), amap_mode="mul")
            if amap.ndim == 2:
                amap = amap[None, ...]
            patch_score_chunks.append(amap.reshape(len(frame_ids), -1))
            patch_gt_chunks.append(gt.reshape(len(frame_ids), -1))
            for idx, frame_id in enumerate(frame_ids):
                frame_scores.setdefault(frame_id, []).append(float(amap[idx].max()))
                frame_labels[frame_id] = int(labels[idx])

    return reduce_patch_predictions(frame_scores, frame_labels, patch_score_chunks, patch_gt_chunks)


def fit_teacher_student(
    config: ProtocolConfig,
    datasets: dict,
    *,
    method_name: str,
    lr: float,
    epochs: int,
    loss_mode: str,
    args,
) -> dict:
    device = torch.device(config.device)
    amp_dtype = precision_to_dtype(config.precision)
    teacher = freeze_module(ResNet50Encoder(pretrained=True).to(device))
    student = ResNet50Decoder(pretrained=False).to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    train_loader = build_loader(datasets["train"], batch_size=config.batch_size, num_workers=config.num_workers, device=device, shuffle=True)
    val_loader = build_loader(datasets["val"], batch_size=config.eval_batch_size, num_workers=config.num_workers, device=device)
    test_loader = build_loader(datasets["test"], batch_size=config.eval_batch_size, num_workers=config.num_workers, device=device)

    best_state = None
    best_val = math.inf
    val_history = []
    train_epochs = 1 if config.smoke else epochs
    for epoch in range(train_epochs):
        student.train()
        losses = []
        for batch in train_loader:
            image = batch["image"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                with device_autocast(device, amp_dtype):
                    feat_t = teacher(image)
            with device_autocast(device, amp_dtype):
                feat_s = student(feat_t)
                if loss_mode == "rd4ad":
                    loss = loss_distil(feat_s, feat_t)
                else:
                    loss = loss_l2(feat_s, feat_t)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        student.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                image = batch["image"].to(device, non_blocking=True)
                with device_autocast(device, amp_dtype):
                    feat_t = teacher(image)
                    feat_s = student(feat_t)
                    if loss_mode == "rd4ad":
                        val_loss = loss_distil(feat_s, feat_t)
                    else:
                        val_loss = loss_l2(feat_s, feat_t)
                val_losses.append(float(val_loss.detach().cpu().item()))
        mean_val = float(np.mean(val_losses)) if val_losses else math.inf
        val_history.append(mean_val)
        if mean_val < best_val:
            best_val = mean_val
            best_state = {k: v.detach().cpu() for k, v in student.state_dict().items()}

    if best_state is not None:
        student.load_state_dict(best_state)
    image_scores, image_labels, frame_ids, pixel_scores, pixel_labels = evaluate_teacher_student(
        teacher,
        student,
        test_loader,
        device=device,
        img_size=config.img_size,
        amp_dtype=amp_dtype,
    )
    image_auroc = compute_safe_auroc(image_labels, image_scores)
    pixel_auroc = compute_safe_auroc(pixel_labels.astype(np.int64), pixel_scores)
    result_json, scores_csv = default_result_paths(config)
    write_scores_csv(scores_csv, frame_ids, image_labels, image_scores)
    if config.save_ckpt and best_state is not None:
        ckpt_dir = config.output_dir()
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": best_state}, ckpt_dir / "best.pt")
    payload = make_method_result(
        config=config,
        image_auroc=image_auroc,
        pixel_auroc=pixel_auroc,
        train_summary=dataset_summary(datasets["train"]),
        val_summary=dataset_summary(datasets["val"]),
        test_summary=dataset_summary(datasets["test"]),
        scores_csv=scores_csv,
        extra={
            "epochs": int(train_epochs),
            "lr": float(lr),
            "best_val_loss": None if best_val == math.inf else float(best_val),
            "val_history": val_history,
            "train_loss_mode": loss_mode,
        },
    )
    write_result_json(result_json, payload)
    payload["result_json"] = str(result_json)
    return payload


class FastFlowHead(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        hidden = max(64, channels // 2)
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def fit_fastflow(config: ProtocolConfig, datasets: dict, args) -> dict:
    device = torch.device(config.device)
    amp_dtype = precision_to_dtype(config.precision)
    teacher = freeze_module(ResNet50Encoder(pretrained=True).to(device))
    head = FastFlowHead(1024).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=args.fastflow_lr)
    train_loader = build_loader(datasets["train"], batch_size=config.batch_size, num_workers=config.num_workers, device=device, shuffle=True)
    val_loader = build_loader(datasets["val"], batch_size=config.eval_batch_size, num_workers=config.num_workers, device=device)
    test_loader = build_loader(datasets["test"], batch_size=config.eval_batch_size, num_workers=config.num_workers, device=device)

    best_state = None
    best_val = math.inf
    epochs = 1 if config.smoke else args.fastflow_epochs
    for _ in range(epochs):
        head.train()
        for batch in train_loader:
            image = batch["image"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                with device_autocast(device, amp_dtype):
                    feat = teacher(image)[-1]
            with device_autocast(device, amp_dtype):
                residual = head(feat)
                loss = (residual ** 2).mean()
            loss.backward()
            optimizer.step()
        head.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                image = batch["image"].to(device, non_blocking=True)
                with device_autocast(device, amp_dtype):
                    feat = teacher(image)[-1]
                    residual = head(feat)
                    val_losses.append(float((residual ** 2).mean().detach().cpu().item()))
        mean_val = float(np.mean(val_losses)) if val_losses else math.inf
        if mean_val < best_val:
            best_val = mean_val
            best_state = {k: v.detach().cpu() for k, v in head.state_dict().items()}
    if best_state is not None:
        head.load_state_dict(best_state)

    frame_scores: dict[str, list[float]] = {}
    frame_labels: dict[str, int] = {}
    patch_score_chunks: list[np.ndarray] = []
    patch_gt_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for batch in test_loader:
            image = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].cpu().numpy().astype(np.int64)
            gt = batch["gt"].cpu().numpy().astype(np.float32)
            frame_ids = list(batch["frame_id"])
            with device_autocast(device, amp_dtype):
                feat = teacher(image)[-1]
                residual = head(feat)
                score_map = (residual.float() ** 2).mean(dim=1, keepdim=True)
            score_map = F.interpolate(score_map, size=(config.img_size, config.img_size), mode="bilinear", align_corners=False)
            score_map = score_map.squeeze(1).detach().cpu().float().numpy()
            patch_score_chunks.append(score_map.reshape(len(frame_ids), -1))
            patch_gt_chunks.append(gt.reshape(len(frame_ids), -1))
            for idx, frame_id in enumerate(frame_ids):
                frame_scores.setdefault(frame_id, []).append(float(score_map[idx].max()))
                frame_labels[frame_id] = int(labels[idx])

    image_scores, image_labels, frame_ids, pixel_scores, pixel_labels = reduce_patch_predictions(
        frame_scores, frame_labels, patch_score_chunks, patch_gt_chunks
    )
    image_auroc = compute_safe_auroc(image_labels, image_scores)
    pixel_auroc = compute_safe_auroc(pixel_labels.astype(np.int64), pixel_scores)
    result_json, scores_csv = default_result_paths(config)
    write_scores_csv(scores_csv, frame_ids, image_labels, image_scores)
    if config.save_ckpt and best_state is not None:
        ckpt_dir = config.output_dir()
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": best_state}, ckpt_dir / "best.pt")
    payload = make_method_result(
        config=config,
        image_auroc=image_auroc,
        pixel_auroc=pixel_auroc,
        train_summary=dataset_summary(datasets["train"]),
        val_summary=dataset_summary(datasets["val"]),
        test_summary=dataset_summary(datasets["test"]),
        scores_csv=scores_csv,
        extra={
            "epochs": int(epochs),
            "lr": float(args.fastflow_lr),
            "best_val_nll": None if best_val == math.inf else float(best_val),
        },
    )
    write_result_json(result_json, payload)
    payload["result_json"] = str(result_json)
    return payload


class DRAEMRecon(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DRAEMSeg(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _perlin_like_mask(batch: int, size: int, device: torch.device) -> torch.Tensor:
    base = torch.rand(batch, 1, max(4, size // 16), max(4, size // 16), device=device)
    base = F.interpolate(base, size=(size, size), mode="bilinear", align_corners=False)
    mask = (base > 0.55).float()
    return mask


def _synthesize_draem_batch(images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch, _, h, w = images.shape
    mask = _perlin_like_mask(batch, h, images.device)
    rolled = torch.roll(images, shifts=max(1, w // 8), dims=-1)
    anomalous = images * (1.0 - mask) + rolled * mask
    return anomalous, mask


def fit_draem(config: ProtocolConfig, datasets: dict, args) -> dict:
    device = torch.device(config.device)
    amp_dtype = precision_to_dtype(config.precision)
    recon = DRAEMRecon().to(device)
    seg = DRAEMSeg().to(device)
    optimizer = torch.optim.Adam(list(recon.parameters()) + list(seg.parameters()), lr=args.draem_lr)
    train_loader = build_loader(datasets["train"], batch_size=config.batch_size, num_workers=config.num_workers, device=device, shuffle=True)
    val_loader = build_loader(datasets["val"], batch_size=config.eval_batch_size, num_workers=config.num_workers, device=device)
    test_loader = build_loader(datasets["test"], batch_size=config.eval_batch_size, num_workers=config.num_workers, device=device)

    best_state = None
    best_val = math.inf
    epochs = 1 if config.smoke else args.draem_epochs
    bce = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        recon.train()
        seg.train()
        for batch in train_loader:
            image = batch["image"].to(device, non_blocking=True)
            anomalous, mask = _synthesize_draem_batch(image)
            optimizer.zero_grad(set_to_none=True)
            with device_autocast(device, amp_dtype):
                recon_img = recon(anomalous)
                seg_logits = seg(torch.cat([anomalous, recon_img], dim=1))
                loss = F.l1_loss(recon_img, image) + bce(seg_logits, mask)
            loss.backward()
            optimizer.step()
        recon.eval()
        seg.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                image = batch["image"].to(device, non_blocking=True)
                anomalous, mask = _synthesize_draem_batch(image)
                with device_autocast(device, amp_dtype):
                    recon_img = recon(anomalous)
                    seg_logits = seg(torch.cat([anomalous, recon_img], dim=1))
                    val_loss = F.l1_loss(recon_img, image) + bce(seg_logits, mask)
                val_losses.append(float(val_loss.detach().cpu().item()))
        mean_val = float(np.mean(val_losses)) if val_losses else math.inf
        if mean_val < best_val:
            best_val = mean_val
            best_state = {
                "recon": {k: v.detach().cpu() for k, v in recon.state_dict().items()},
                "seg": {k: v.detach().cpu() for k, v in seg.state_dict().items()},
            }
    if best_state is not None:
        recon.load_state_dict(best_state["recon"])
        seg.load_state_dict(best_state["seg"])

    frame_scores: dict[str, list[float]] = {}
    frame_labels: dict[str, int] = {}
    patch_score_chunks: list[np.ndarray] = []
    patch_gt_chunks: list[np.ndarray] = []
    recon.eval()
    seg.eval()
    with torch.no_grad():
        for batch in test_loader:
            image = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].cpu().numpy().astype(np.int64)
            gt = batch["gt"].cpu().numpy().astype(np.float32)
            frame_ids = list(batch["frame_id"])
            with device_autocast(device, amp_dtype):
                recon_img = recon(image)
                seg_logits = seg(torch.cat([image, recon_img], dim=1))
                score_map = torch.sigmoid(seg_logits).squeeze(1)
            score_map = score_map.detach().cpu().float().numpy()
            patch_score_chunks.append(score_map.reshape(len(frame_ids), -1))
            patch_gt_chunks.append(gt.reshape(len(frame_ids), -1))
            for idx, frame_id in enumerate(frame_ids):
                frame_scores.setdefault(frame_id, []).append(float(score_map[idx].max()))
                frame_labels[frame_id] = int(labels[idx])

    image_scores, image_labels, frame_ids, pixel_scores, pixel_labels = reduce_patch_predictions(
        frame_scores, frame_labels, patch_score_chunks, patch_gt_chunks
    )
    image_auroc = compute_safe_auroc(image_labels, image_scores)
    pixel_auroc = compute_safe_auroc(pixel_labels.astype(np.int64), pixel_scores)
    result_json, scores_csv = default_result_paths(config)
    write_scores_csv(scores_csv, frame_ids, image_labels, image_scores)
    if config.save_ckpt and best_state is not None:
        ckpt_dir = config.output_dir()
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, ckpt_dir / "best.pt")
    payload = make_method_result(
        config=config,
        image_auroc=image_auroc,
        pixel_auroc=pixel_auroc,
        train_summary=dataset_summary(datasets["train"]),
        val_summary=dataset_summary(datasets["val"]),
        test_summary=dataset_summary(datasets["test"]),
        scores_csv=scores_csv,
        extra={
            "epochs": int(epochs),
            "lr": float(args.draem_lr),
            "best_val_loss": None if best_val == math.inf else float(best_val),
            "synthetic_anomaly": "perlin_self_cutpaste",
        },
    )
    write_result_json(result_json, payload)
    payload["result_json"] = str(result_json)
    return payload


def build_config(args, method: str) -> ProtocolConfig:
    batch_size, eval_batch_size = resolve_method_batch_sizes(args, method)
    return ProtocolConfig(
        method=method,
        train_root=args.train_root,
        test_root=args.test_root,
        view_id=args.view_id,
        img_size=args.img_size,
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        train_sample_num=args.train_sample_num,
        train_val_test_split=tuple(args.train_val_test_split),
        sampling_mode=args.sampling_mode,
        device=args.device,
        precision=args.precision,
        seed=args.seed,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=args.num_workers,
        preload=args.preload,
        preload_workers=args.preload_workers,
        smoke=args.smoke,
        smoke_train_images=args.smoke_train_images,
        smoke_train_img_size=args.smoke_train_img_size,
        save_ckpt=args.save_ckpt,
        save_maps=args.save_maps,
        export_bank=args.export_bank,
        output_root=args.output_root,
        results_root=args.results_root,
        result_json_path=args.result_json,
        scores_csv_path=args.scores_csv,
    )


def run_single_method(method: str, args) -> dict:
    config = build_config(args, method)
    seed_everything(config.seed)
    datasets = build_protocol_datasets(config)
    if method == "patchcore":
        return evaluate_patchcore(config, datasets, args)
    if method == "padim":
        return evaluate_padim(config, datasets, args)
    if method == "rd4ad_i":
        return fit_teacher_student(config, datasets, method_name=method, lr=args.rd4ad_lr, epochs=args.rd4ad_epochs, loss_mode="rd4ad", args=args)
    if method == "stfpm":
        return fit_teacher_student(config, datasets, method_name=method, lr=args.stfpm_lr, epochs=args.stfpm_epochs, loss_mode="stfpm", args=args)
    if method == "fastflow":
        return fit_fastflow(config, datasets, args)
    if method == "draem":
        return fit_draem(config, datasets, args)
    raise ValueError(f"Unsupported method: {method}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified RGB-only baseline launcher")
    parser.add_argument("--method", type=str, default="all", choices=[*METHODS, "all"])
    parser.add_argument("--train_root", type=str, default="/data1/Leaddo_data/20260327-resize512")
    parser.add_argument("--test_root", type=str, default="/home/root123/LF/WYM/SteelRailWay/rail_mvtec_gt_test")
    parser.add_argument("--view_id", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--train_sample_num", type=int, default=1500)
    parser.add_argument("--train_val_test_split", type=float, nargs=3, default=[0.9, 0.1, 0.0])
    parser.add_argument("--sampling_mode", type=str, default="uniform_time", choices=["uniform_time", "random"])
    parser.add_argument("--patch_size", type=int, default=900)
    parser.add_argument("--patch_stride", type=int, default=850)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--eval_batch_size", type=int, default=-1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--preload", action="store_true")
    parser.add_argument("--preload_workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke_train_images", type=int, default=16)
    parser.add_argument("--smoke_train_img_size", type=int, default=256)
    parser.add_argument("--save_ckpt", action="store_true")
    parser.add_argument("--save_maps", action="store_true")
    parser.add_argument("--export_bank", action="store_true")
    parser.add_argument("--output_root", type=str, default="")
    parser.add_argument("--results_root", type=str, default="")
    parser.add_argument("--result_json", type=str, default="")
    parser.add_argument("--scores_csv", type=str, default="")
    parser.add_argument("--summary_csv", type=str, default="")
    parser.add_argument("--summary_json", type=str, default="")

    parser.add_argument("--patchcore_layers", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--patchcore_target_map_size", type=int, default=32)
    parser.add_argument("--patchcore_projection_dim", type=int, default=128)
    parser.add_argument("--patchcore_coreset_ratio", type=float, default=0.01)
    parser.add_argument("--patchcore_coreset_size", type=int, default=0)
    parser.add_argument("--patchcore_k", type=int, default=5)
    parser.add_argument("--patchcore_query_chunk_size", type=int, default=2048)

    parser.add_argument("--padim_layers", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--padim_target_map_size", type=int, default=16)

    parser.add_argument("--rd4ad_epochs", type=int, default=20)
    parser.add_argument("--rd4ad_lr", type=float, default=1e-4)
    parser.add_argument("--stfpm_epochs", type=int, default=20)
    parser.add_argument("--stfpm_lr", type=float, default=1e-4)
    parser.add_argument("--fastflow_epochs", type=int, default=20)
    parser.add_argument("--fastflow_lr", type=float, default=1e-4)
    parser.add_argument("--draem_epochs", type=int, default=20)
    parser.add_argument("--draem_lr", type=float, default=1e-4)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    methods = list(METHODS) if args.method == "all" else [args.method]
    summary_rows = []
    for method in methods:
        print("=" * 72)
        print(f"[RUN] method={method} view_id={args.view_id}")
        print("=" * 72)
        payload = run_single_method(method, args)
        summary_rows.append(
            {
                "method": payload["method"],
                "view_id": payload["view_id"],
                "image_auroc": payload["image_auroc"],
                "pixel_auroc": payload["pixel_auroc"],
                "smoke": payload["smoke"],
                "train_images": payload["train_summary"]["num_images"],
                "val_images": payload["val_summary"]["num_images"],
                "test_images": payload["test_summary"]["num_images"],
                "result_json": payload["result_json"],
            }
        )
    if args.method == "all":
        if args.summary_csv:
            summary_csv = Path(args.summary_csv)
        elif args.smoke:
            summary_csv = PROJ_ROOT / "$out" / "baselines_rgb_smoke" / f"cam{args.view_id}" / "summary.csv"
        else:
            summary_csv = PROJ_ROOT / "results" / "baselines_rgb" / f"cam{args.view_id}" / "summary.csv"
        if args.summary_json:
            summary_json = Path(args.summary_json)
        elif args.smoke:
            summary_json = PROJ_ROOT / "$out" / "baselines_rgb_smoke" / f"cam{args.view_id}" / "summary.json"
        else:
            summary_json = PROJ_ROOT / "results" / "baselines_rgb" / f"cam{args.view_id}" / "summary.json"
        write_summary_files(summary_rows, summary_csv, summary_json)
        print(f"[DONE] summary_csv={summary_csv}")
        print(f"[DONE] summary_json={summary_json}")


if __name__ == "__main__":
    main()
