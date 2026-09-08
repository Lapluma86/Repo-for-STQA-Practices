#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unified RGB+Depth baseline launcher for aligned dual-modal comparison experiments."""

from __future__ import annotations

import argparse
import math
import sys
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
    FUSION_SOURCES,
    ProtocolConfig,
    build_loader,
    build_protocol_datasets,
    compute_source_aurocs,
    dataset_summary,
    default_scores_csv_by_source,
    make_method_result,
    precision_to_dtype,
    reduce_multisource_patch_predictions,
    seed_everything,
    write_result_json,
    write_scores_csv,
    write_summary_files,
)
from scripts.baselines.run_rgb_baseline import (
    DRAEMRecon,
    DRAEMSeg,
    FastFlowHead,
    METHOD_BATCH_DEFAULTS,
    batched_knn_distances,
    device_autocast,
    extract_embeddings,
    freeze_module,
    parse_layers,
    resolve_method_batch_sizes,
)
from utils.losses import loss_distil, loss_l2


METHODS = ("padim", "patchcore", "fastflow", "draem", "stfpm", "rd4ad_i")


def batch_modal_tensor(batch: dict, modality: str) -> torch.Tensor:
    if modality == "rgb":
        return batch["image"]
    if modality == "depth":
        return batch["depth"]
    raise KeyError(f"Unsupported modality: {modality}")


def collect_memory_features_modal(
    encoder: nn.Module,
    loader,
    *,
    modality: str,
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
            image = batch_modal_tensor(batch, modality).to(device, non_blocking=True)
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
        raise RuntimeError(f"No memory-bank features were extracted for modality={modality}")
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


def collect_layer_features_modal(
    encoder: nn.Module,
    loader,
    *,
    modality: str,
    device: torch.device,
    layers: tuple[int, ...],
    target_map_size: int,
    amp_dtype: torch.dtype | None,
) -> dict[int, np.ndarray]:
    encoder.eval()
    collected: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
    with torch.no_grad():
        for batch in loader:
            image = batch_modal_tensor(batch, modality).to(device, non_blocking=True)
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


def synthesize_multisource_result(
    config: ProtocolConfig,
    datasets: dict,
    reduced: dict,
    args,
    *,
    extra: dict | None = None,
) -> dict:
    selected_source = args.score_source
    image_auroc_by_source, pixel_auroc_by_source = compute_source_aurocs(
        reduced["image_labels"],
        reduced["image_scores_by_source"],
        reduced["pixel_labels"],
        reduced["pixel_scores_by_source"],
    )
    result_json, scores_csv_by_source = default_scores_csv_by_source(
        config,
        reduced["source_order"],
        selected_source=selected_source,
    )
    for source in reduced["source_order"]:
        write_scores_csv(
            scores_csv_by_source[source],
            reduced["frame_ids"],
            reduced["image_labels"],
            reduced["image_scores_by_source"][source],
        )
    payload_extra = {
        "score_source_selected": selected_source,
        "fusion_rule": args.fusion_rule,
        "image_auroc_by_source": image_auroc_by_source,
        "pixel_auroc_by_source": pixel_auroc_by_source,
        "scores_csv_by_source": {source: str(path) for source, path in scores_csv_by_source.items()},
        "pixel_stats": reduced["pixel_stats"],
        **reduced.get("fusion_meta", {}),
    }
    if extra:
        payload_extra.update(extra)
    payload = make_method_result(
        config=config,
        image_auroc=image_auroc_by_source[selected_source],
        pixel_auroc=pixel_auroc_by_source[selected_source],
        train_summary=dataset_summary(datasets["train"]),
        val_summary=dataset_summary(datasets["val"]),
        test_summary=dataset_summary(datasets["test"]),
        scores_csv=scores_csv_by_source[selected_source],
        extra=payload_extra,
    )
    write_result_json(result_json, payload)
    payload["result_json"] = str(result_json)
    return payload


def evaluate_patchcore_rgbd(config: ProtocolConfig, datasets: dict, args) -> dict:
    device = torch.device(config.device)
    amp_dtype = precision_to_dtype(config.precision)
    encoder = freeze_module(ResNet50Encoder(pretrained=True).to(device))
    train_loader = build_loader(
        datasets["train"],
        batch_size=config.eval_batch_size,
        num_workers=config.num_workers,
        device=device,
    )
    test_loader = build_loader(
        datasets["test"],
        batch_size=config.eval_batch_size,
        num_workers=config.num_workers,
        device=device,
    )
    layers = parse_layers(args.patchcore_layers)

    branch_meta = {}
    indexes = {}
    for modality in ("rgb", "depth"):
        memory_full, embedding_dim = collect_memory_features_modal(
            encoder,
            train_loader,
            modality=modality,
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
        indexes[modality] = (index, projector)
        branch_meta[modality] = {
            "embedding_dim_before_projection": int(embedding_dim),
            "embedding_dim_after_projection": int(memory_bank.shape[1]),
            "memory_vectors_full": int(memory_full.shape[0]),
            "memory_vectors_coreset": int(memory_bank.shape[0]),
            "memory_coreset_indices_count": int(len(coreset_indices)),
        }

    patch_frame_ids: list[str] = []
    patch_labels: list[int] = []
    patch_score_chunks_by_source: dict[str, list[np.ndarray]] = {"rgb": [], "depth": []}
    patch_gt_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for batch in test_loader:
            labels = batch["label"].cpu().numpy().astype(np.int64)
            gt = batch["gt"].cpu().numpy().astype(np.float32)
            frame_ids = list(batch["frame_id"])
            patch_frame_ids.extend(frame_ids)
            patch_labels.extend(labels.tolist())
            patch_gt_chunks.append(gt)
            for modality in ("rgb", "depth"):
                image = batch_modal_tensor(batch, modality).to(device, non_blocking=True)
                with device_autocast(device, amp_dtype):
                    embedding, _, h, w = extract_embeddings(
                        encoder,
                        image,
                        layers=layers,
                        target_map_size=args.patchcore_target_map_size,
                    )
                queries = embedding.reshape(-1, embedding.shape[-1]).detach().float().cpu().numpy().astype(np.float32)
                index, projector = indexes[modality]
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
                ).squeeze(1).numpy().astype(np.float32, copy=False)
                patch_score_chunks_by_source[modality].append(patch_maps)

    reduced = reduce_multisource_patch_predictions(
        patch_frame_ids,
        patch_labels,
        patch_score_chunks_by_source,
        patch_gt_chunks,
        fusion_rule=args.fusion_rule,
        selected_source=args.score_source,
    )
    return synthesize_multisource_result(
        config,
        datasets,
        reduced,
        args,
        extra={
            "layers": list(layers),
            "target_map_size": int(args.patchcore_target_map_size),
            "projection_dim_arg": int(args.patchcore_projection_dim),
            "coreset_ratio": float(args.patchcore_coreset_ratio),
            "coreset_size_arg": int(args.patchcore_coreset_size),
            "k_nn": int(args.patchcore_k),
            "query_chunk_size": int(args.patchcore_query_chunk_size),
            "branch_meta": branch_meta,
        },
    )


def evaluate_padim_rgbd(config: ProtocolConfig, datasets: dict, args) -> dict:
    device = torch.device(config.device)
    amp_dtype = precision_to_dtype(config.precision)
    encoder = freeze_module(ResNet50Encoder(pretrained=True).to(device))
    train_loader = build_loader(
        datasets["train"],
        batch_size=config.eval_batch_size,
        num_workers=config.num_workers,
        device=device,
    )
    test_loader = build_loader(
        datasets["test"],
        batch_size=config.eval_batch_size,
        num_workers=config.num_workers,
        device=device,
    )
    layers = parse_layers(args.padim_layers)
    params_by_modality = {}
    for modality in ("rgb", "depth"):
        train_features = collect_layer_features_modal(
            encoder,
            train_loader,
            modality=modality,
            device=device,
            layers=layers,
            target_map_size=args.padim_target_map_size,
            amp_dtype=amp_dtype,
        )
        params_by_modality[modality] = estimate_gaussians(train_features, diagonal_only=config.smoke)

    patch_frame_ids: list[str] = []
    patch_labels: list[int] = []
    patch_score_chunks_by_source: dict[str, list[np.ndarray]] = {"rgb": [], "depth": []}
    patch_gt_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for batch in test_loader:
            labels = batch["label"].cpu().numpy().astype(np.int64)
            gt = batch["gt"].cpu().numpy().astype(np.float32)
            frame_ids = list(batch["frame_id"])
            patch_frame_ids.extend(frame_ids)
            patch_labels.extend(labels.tolist())
            patch_gt_chunks.append(gt)
            for modality in ("rgb", "depth"):
                image = batch_modal_tensor(batch, modality).to(device, non_blocking=True)
                with device_autocast(device, amp_dtype):
                    feats = encoder(image)
                batch_maps = []
                for layer_idx in layers:
                    feat = feats[layer_idx]
                    if args.padim_target_map_size > 0 and feat.shape[-1] != args.padim_target_map_size:
                        feat = F.adaptive_avg_pool2d(
                            feat,
                            (args.padim_target_map_size, args.padim_target_map_size),
                        )
                    feat_np = feat.detach().float().cpu().numpy().astype(np.float32)
                    bsz, channels, h, w = feat_np.shape
                    means, inv_covs = params_by_modality[modality][layer_idx]
                    feat_patches = feat_np.transpose(0, 2, 3, 1).reshape(bsz, h * w, channels)
                    scores = np.zeros((bsz, h * w), dtype=np.float32)
                    diagonal_only = inv_covs.ndim == 2
                    for b in range(bsz):
                        for patch_idx in range(h * w):
                            diff = feat_patches[b, patch_idx] - means[patch_idx]
                            if diagonal_only:
                                scores[b, patch_idx] = float(
                                    np.sqrt((diff * diff * inv_covs[patch_idx]).sum())
                                )
                            else:
                                scores[b, patch_idx] = float(
                                    np.sqrt(diff @ inv_covs[patch_idx] @ diff)
                                )
                    batch_maps.append(scores.reshape(bsz, h, w))
                combined = np.max(np.stack(batch_maps, axis=0), axis=0)
                combined = torch.from_numpy(combined).unsqueeze(1)
                combined = F.interpolate(
                    combined,
                    size=(config.img_size, config.img_size),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1).numpy().astype(np.float32, copy=False)
                patch_score_chunks_by_source[modality].append(combined)

    reduced = reduce_multisource_patch_predictions(
        patch_frame_ids,
        patch_labels,
        patch_score_chunks_by_source,
        patch_gt_chunks,
        fusion_rule=args.fusion_rule,
        selected_source=args.score_source,
    )
    return synthesize_multisource_result(
        config,
        datasets,
        reduced,
        args,
        extra={
            "layers": list(layers),
            "target_map_size": int(args.padim_target_map_size),
            "covariance_mode": "diagonal_smoke" if config.smoke else "full",
        },
    )


def dual_feature_distance_map(
    teacher: nn.Module,
    student_rgb: nn.Module,
    student_depth: nn.Module,
    batch: dict,
    *,
    device: torch.device,
    img_size: int,
    amp_dtype: torch.dtype | None,
) -> tuple[np.ndarray, np.ndarray]:
    rgb = batch["image"].to(device, non_blocking=True)
    depth = batch["depth"].to(device, non_blocking=True)
    with device_autocast(device, amp_dtype):
        feat_t_rgb = teacher(rgb)
        feat_t_depth = teacher(depth)
        feat_s_rgb = student_rgb(feat_t_rgb)
        feat_s_depth = student_depth(feat_t_depth)
    amap_rgb, _ = cal_anomaly_map(feat_s_rgb, feat_t_rgb, out_size=(img_size, img_size), amap_mode="mul")
    amap_depth, _ = cal_anomaly_map(
        feat_s_depth,
        feat_t_depth,
        out_size=(img_size, img_size),
        amap_mode="mul",
    )
    if np.asarray(amap_rgb).ndim == 2:
        amap_rgb = np.asarray(amap_rgb)[None, ...]
    if np.asarray(amap_depth).ndim == 2:
        amap_depth = np.asarray(amap_depth)[None, ...]
    return (
        np.asarray(amap_rgb, dtype=np.float32),
        np.asarray(amap_depth, dtype=np.float32),
    )


def evaluate_teacher_student_rgbd(
    teacher: nn.Module,
    student_rgb: nn.Module,
    student_depth: nn.Module,
    loader,
    *,
    device: torch.device,
    img_size: int,
    amp_dtype: torch.dtype | None,
    fusion_rule: str,
    selected_source: str,
) -> dict:
    teacher.eval()
    student_rgb.eval()
    student_depth.eval()
    patch_frame_ids: list[str] = []
    patch_labels: list[int] = []
    patch_gt_chunks: list[np.ndarray] = []
    patch_score_chunks_by_source: dict[str, list[np.ndarray]] = {"rgb": [], "depth": []}
    with torch.no_grad():
        for batch in loader:
            labels = batch["label"].cpu().numpy().astype(np.int64)
            gt = batch["gt"].cpu().numpy().astype(np.float32)
            frame_ids = list(batch["frame_id"])
            patch_frame_ids.extend(frame_ids)
            patch_labels.extend(labels.tolist())
            patch_gt_chunks.append(gt)
            score_rgb, score_depth = dual_feature_distance_map(
                teacher,
                student_rgb,
                student_depth,
                batch,
                device=device,
                img_size=img_size,
                amp_dtype=amp_dtype,
            )
            patch_score_chunks_by_source["rgb"].append(score_rgb)
            patch_score_chunks_by_source["depth"].append(score_depth)
    return reduce_multisource_patch_predictions(
        patch_frame_ids,
        patch_labels,
        patch_score_chunks_by_source,
        patch_gt_chunks,
        fusion_rule=fusion_rule,
        selected_source=selected_source,
    )


def fit_teacher_student_rgbd(
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
    student_rgb = ResNet50Decoder(pretrained=False).to(device)
    student_depth = ResNet50Decoder(pretrained=False).to(device)
    optimizer = torch.optim.Adam(
        list(student_rgb.parameters()) + list(student_depth.parameters()),
        lr=lr,
    )
    train_loader = build_loader(
        datasets["train"],
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        device=device,
        shuffle=True,
    )
    val_loader = build_loader(
        datasets["val"],
        batch_size=config.eval_batch_size,
        num_workers=config.num_workers,
        device=device,
    )
    test_loader = build_loader(
        datasets["test"],
        batch_size=config.eval_batch_size,
        num_workers=config.num_workers,
        device=device,
    )

    best_state = None
    best_val = math.inf
    val_history = []
    train_epochs = 1 if config.smoke else epochs
    for _ in range(train_epochs):
        student_rgb.train()
        student_depth.train()
        for batch in train_loader:
            rgb = batch["image"].to(device, non_blocking=True)
            depth = batch["depth"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                with device_autocast(device, amp_dtype):
                    feat_t_rgb = teacher(rgb)
                    feat_t_depth = teacher(depth)
            with device_autocast(device, amp_dtype):
                feat_s_rgb = student_rgb(feat_t_rgb)
                feat_s_depth = student_depth(feat_t_depth)
                if loss_mode == "rd4ad":
                    loss_rgb = loss_distil(feat_s_rgb, feat_t_rgb)
                    loss_depth = loss_distil(feat_s_depth, feat_t_depth)
                else:
                    loss_rgb = loss_l2(feat_s_rgb, feat_t_rgb)
                    loss_depth = loss_l2(feat_s_depth, feat_t_depth)
                loss = loss_rgb + loss_depth
            loss.backward()
            optimizer.step()
        reduced_val = evaluate_teacher_student_rgbd(
            teacher,
            student_rgb,
            student_depth,
            val_loader,
            device=device,
            img_size=config.img_size,
            amp_dtype=amp_dtype,
            fusion_rule=args.fusion_rule,
            selected_source="fusion",
        )
        mean_val = float(np.mean(reduced_val["image_scores_by_source"]["fusion"])) if reduced_val["frame_ids"] else math.inf
        val_history.append(mean_val)
        if mean_val < best_val:
            best_val = mean_val
            best_state = {
                "student_rgb": {k: v.detach().cpu() for k, v in student_rgb.state_dict().items()},
                "student_depth": {k: v.detach().cpu() for k, v in student_depth.state_dict().items()},
            }

    if best_state is not None:
        student_rgb.load_state_dict(best_state["student_rgb"])
        student_depth.load_state_dict(best_state["student_depth"])
    reduced = evaluate_teacher_student_rgbd(
        teacher,
        student_rgb,
        student_depth,
        test_loader,
        device=device,
        img_size=config.img_size,
        amp_dtype=amp_dtype,
        fusion_rule=args.fusion_rule,
        selected_source=args.score_source,
    )
    if config.save_ckpt and best_state is not None:
        ckpt_dir = config.output_dir()
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, ckpt_dir / "best.pt")
    return synthesize_multisource_result(
        config,
        datasets,
        reduced,
        args,
        extra={
            "epochs": int(train_epochs),
            "lr": float(lr),
            "best_val_fusion_mean": None if best_val == math.inf else float(best_val),
            "val_history": val_history,
            "train_loss_mode": loss_mode,
            "teacher_shared": True,
        },
    )


def evaluate_fastflow_rgbd(
    teacher: nn.Module,
    head_rgb: nn.Module,
    head_depth: nn.Module,
    loader,
    *,
    device: torch.device,
    img_size: int,
    amp_dtype: torch.dtype | None,
    fusion_rule: str,
    selected_source: str,
) -> dict:
    teacher.eval()
    head_rgb.eval()
    head_depth.eval()
    patch_frame_ids: list[str] = []
    patch_labels: list[int] = []
    patch_gt_chunks: list[np.ndarray] = []
    patch_score_chunks_by_source: dict[str, list[np.ndarray]] = {"rgb": [], "depth": []}
    with torch.no_grad():
        for batch in loader:
            rgb = batch["image"].to(device, non_blocking=True)
            depth = batch["depth"].to(device, non_blocking=True)
            labels = batch["label"].cpu().numpy().astype(np.int64)
            gt = batch["gt"].cpu().numpy().astype(np.float32)
            frame_ids = list(batch["frame_id"])
            patch_frame_ids.extend(frame_ids)
            patch_labels.extend(labels.tolist())
            patch_gt_chunks.append(gt)
            with device_autocast(device, amp_dtype):
                feat_rgb = teacher(rgb)[-1]
                feat_depth = teacher(depth)[-1]
                residual_rgb = head_rgb(feat_rgb)
                residual_depth = head_depth(feat_depth)
                score_rgb = (residual_rgb.float() ** 2).mean(dim=1, keepdim=True)
                score_depth = (residual_depth.float() ** 2).mean(dim=1, keepdim=True)
            score_rgb = F.interpolate(
                score_rgb,
                size=(img_size, img_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1).detach().cpu().float().numpy().astype(np.float32, copy=False)
            score_depth = F.interpolate(
                score_depth,
                size=(img_size, img_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1).detach().cpu().float().numpy().astype(np.float32, copy=False)
            patch_score_chunks_by_source["rgb"].append(score_rgb)
            patch_score_chunks_by_source["depth"].append(score_depth)
    return reduce_multisource_patch_predictions(
        patch_frame_ids,
        patch_labels,
        patch_score_chunks_by_source,
        patch_gt_chunks,
        fusion_rule=fusion_rule,
        selected_source=selected_source,
    )


def fit_fastflow_rgbd(config: ProtocolConfig, datasets: dict, args) -> dict:
    device = torch.device(config.device)
    amp_dtype = precision_to_dtype(config.precision)
    teacher = freeze_module(ResNet50Encoder(pretrained=True).to(device))
    head_rgb = FastFlowHead(1024).to(device)
    head_depth = FastFlowHead(1024).to(device)
    optimizer = torch.optim.Adam(
        list(head_rgb.parameters()) + list(head_depth.parameters()),
        lr=args.fastflow_lr,
    )
    train_loader = build_loader(
        datasets["train"],
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        device=device,
        shuffle=True,
    )
    val_loader = build_loader(
        datasets["val"],
        batch_size=config.eval_batch_size,
        num_workers=config.num_workers,
        device=device,
    )
    test_loader = build_loader(
        datasets["test"],
        batch_size=config.eval_batch_size,
        num_workers=config.num_workers,
        device=device,
    )

    best_state = None
    best_val = math.inf
    val_history = []
    train_epochs = 1 if config.smoke else args.fastflow_epochs
    for _ in range(train_epochs):
        head_rgb.train()
        head_depth.train()
        for batch in train_loader:
            rgb = batch["image"].to(device, non_blocking=True)
            depth = batch["depth"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                with device_autocast(device, amp_dtype):
                    feat_rgb = teacher(rgb)[-1]
                    feat_depth = teacher(depth)[-1]
            with device_autocast(device, amp_dtype):
                residual_rgb = head_rgb(feat_rgb)
                residual_depth = head_depth(feat_depth)
                loss = (residual_rgb ** 2).mean() + (residual_depth ** 2).mean()
            loss.backward()
            optimizer.step()
        reduced_val = evaluate_fastflow_rgbd(
            teacher,
            head_rgb,
            head_depth,
            val_loader,
            device=device,
            img_size=config.img_size,
            amp_dtype=amp_dtype,
            fusion_rule=args.fusion_rule,
            selected_source="fusion",
        )
        mean_val = float(np.mean(reduced_val["image_scores_by_source"]["fusion"])) if reduced_val["frame_ids"] else math.inf
        val_history.append(mean_val)
        if mean_val < best_val:
            best_val = mean_val
            best_state = {
                "head_rgb": {k: v.detach().cpu() for k, v in head_rgb.state_dict().items()},
                "head_depth": {k: v.detach().cpu() for k, v in head_depth.state_dict().items()},
            }
    if best_state is not None:
        head_rgb.load_state_dict(best_state["head_rgb"])
        head_depth.load_state_dict(best_state["head_depth"])

    reduced = evaluate_fastflow_rgbd(
        teacher,
        head_rgb,
        head_depth,
        test_loader,
        device=device,
        img_size=config.img_size,
        amp_dtype=amp_dtype,
        fusion_rule=args.fusion_rule,
        selected_source=args.score_source,
    )
    if config.save_ckpt and best_state is not None:
        ckpt_dir = config.output_dir()
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, ckpt_dir / "best.pt")
    return synthesize_multisource_result(
        config,
        datasets,
        reduced,
        args,
        extra={
            "epochs": int(train_epochs),
            "lr": float(args.fastflow_lr),
            "best_val_fusion_mean": None if best_val == math.inf else float(best_val),
            "val_history": val_history,
            "teacher_shared": True,
        },
    )


def _perlin_like_mask(batch: int, size: int, device: torch.device) -> torch.Tensor:
    base = torch.rand(batch, 1, max(4, size // 16), max(4, size // 16), device=device)
    base = F.interpolate(base, size=(size, size), mode="bilinear", align_corners=False)
    mask = (base > 0.55).float()
    return mask


def _synthesize_draem_pair(
    rgb: torch.Tensor,
    depth: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, _, h, w = rgb.shape
    mask = _perlin_like_mask(batch, h, rgb.device)
    shift = max(1, w // 8)
    rolled_rgb = torch.roll(rgb, shifts=shift, dims=-1)
    rolled_depth = torch.roll(depth, shifts=shift, dims=-1)
    anomalous_rgb = rgb * (1.0 - mask) + rolled_rgb * mask
    anomalous_depth = depth * (1.0 - mask) + rolled_depth * mask
    return anomalous_rgb, anomalous_depth, mask


def evaluate_draem_rgbd(
    recon_rgb: nn.Module,
    seg_rgb: nn.Module,
    recon_depth: nn.Module,
    seg_depth: nn.Module,
    loader,
    *,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    fusion_rule: str,
    selected_source: str,
) -> dict:
    recon_rgb.eval()
    seg_rgb.eval()
    recon_depth.eval()
    seg_depth.eval()
    patch_frame_ids: list[str] = []
    patch_labels: list[int] = []
    patch_gt_chunks: list[np.ndarray] = []
    patch_score_chunks_by_source: dict[str, list[np.ndarray]] = {"rgb": [], "depth": []}
    with torch.no_grad():
        for batch in loader:
            rgb = batch["image"].to(device, non_blocking=True)
            depth = batch["depth"].to(device, non_blocking=True)
            labels = batch["label"].cpu().numpy().astype(np.int64)
            gt = batch["gt"].cpu().numpy().astype(np.float32)
            frame_ids = list(batch["frame_id"])
            patch_frame_ids.extend(frame_ids)
            patch_labels.extend(labels.tolist())
            patch_gt_chunks.append(gt)
            with device_autocast(device, amp_dtype):
                recon_img_rgb = recon_rgb(rgb)
                seg_logits_rgb = seg_rgb(torch.cat([rgb, recon_img_rgb], dim=1))
                recon_img_depth = recon_depth(depth)
                seg_logits_depth = seg_depth(torch.cat([depth, recon_img_depth], dim=1))
                score_rgb = torch.sigmoid(seg_logits_rgb).squeeze(1)
                score_depth = torch.sigmoid(seg_logits_depth).squeeze(1)
            patch_score_chunks_by_source["rgb"].append(
                score_rgb.detach().cpu().float().numpy().astype(np.float32, copy=False)
            )
            patch_score_chunks_by_source["depth"].append(
                score_depth.detach().cpu().float().numpy().astype(np.float32, copy=False)
            )
    return reduce_multisource_patch_predictions(
        patch_frame_ids,
        patch_labels,
        patch_score_chunks_by_source,
        patch_gt_chunks,
        fusion_rule=fusion_rule,
        selected_source=selected_source,
    )


def fit_draem_rgbd(config: ProtocolConfig, datasets: dict, args) -> dict:
    device = torch.device(config.device)
    amp_dtype = precision_to_dtype(config.precision)
    recon_rgb = DRAEMRecon().to(device)
    seg_rgb = DRAEMSeg().to(device)
    recon_depth = DRAEMRecon().to(device)
    seg_depth = DRAEMSeg().to(device)
    optimizer = torch.optim.Adam(
        list(recon_rgb.parameters())
        + list(seg_rgb.parameters())
        + list(recon_depth.parameters())
        + list(seg_depth.parameters()),
        lr=args.draem_lr,
    )
    train_loader = build_loader(
        datasets["train"],
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        device=device,
        shuffle=True,
    )
    val_loader = build_loader(
        datasets["val"],
        batch_size=config.eval_batch_size,
        num_workers=config.num_workers,
        device=device,
    )
    test_loader = build_loader(
        datasets["test"],
        batch_size=config.eval_batch_size,
        num_workers=config.num_workers,
        device=device,
    )

    best_state = None
    best_val = math.inf
    val_history = []
    epochs = 1 if config.smoke else args.draem_epochs
    bce = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        recon_rgb.train()
        seg_rgb.train()
        recon_depth.train()
        seg_depth.train()
        for batch in train_loader:
            rgb = batch["image"].to(device, non_blocking=True)
            depth = batch["depth"].to(device, non_blocking=True)
            anomalous_rgb, anomalous_depth, mask = _synthesize_draem_pair(rgb, depth)
            optimizer.zero_grad(set_to_none=True)
            with device_autocast(device, amp_dtype):
                recon_out_rgb = recon_rgb(anomalous_rgb)
                seg_logits_rgb = seg_rgb(torch.cat([anomalous_rgb, recon_out_rgb], dim=1))
                recon_out_depth = recon_depth(anomalous_depth)
                seg_logits_depth = seg_depth(torch.cat([anomalous_depth, recon_out_depth], dim=1))
                loss = (
                    F.l1_loss(recon_out_rgb, rgb)
                    + bce(seg_logits_rgb, mask)
                    + F.l1_loss(recon_out_depth, depth)
                    + bce(seg_logits_depth, mask)
                )
            loss.backward()
            optimizer.step()
        recon_rgb.eval()
        seg_rgb.eval()
        recon_depth.eval()
        seg_depth.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                rgb = batch["image"].to(device, non_blocking=True)
                depth = batch["depth"].to(device, non_blocking=True)
                anomalous_rgb, anomalous_depth, mask = _synthesize_draem_pair(rgb, depth)
                with device_autocast(device, amp_dtype):
                    recon_out_rgb = recon_rgb(anomalous_rgb)
                    seg_logits_rgb = seg_rgb(torch.cat([anomalous_rgb, recon_out_rgb], dim=1))
                    recon_out_depth = recon_depth(anomalous_depth)
                    seg_logits_depth = seg_depth(torch.cat([anomalous_depth, recon_out_depth], dim=1))
                    val_loss = (
                        F.l1_loss(recon_out_rgb, rgb)
                        + bce(seg_logits_rgb, mask)
                        + F.l1_loss(recon_out_depth, depth)
                        + bce(seg_logits_depth, mask)
                    )
                val_losses.append(float(val_loss.detach().cpu().item()))
        mean_val = float(np.mean(val_losses)) if val_losses else math.inf
        val_history.append(mean_val)
        if mean_val < best_val:
            best_val = mean_val
            best_state = {
                "recon_rgb": {k: v.detach().cpu() for k, v in recon_rgb.state_dict().items()},
                "seg_rgb": {k: v.detach().cpu() for k, v in seg_rgb.state_dict().items()},
                "recon_depth": {k: v.detach().cpu() for k, v in recon_depth.state_dict().items()},
                "seg_depth": {k: v.detach().cpu() for k, v in seg_depth.state_dict().items()},
            }
    if best_state is not None:
        recon_rgb.load_state_dict(best_state["recon_rgb"])
        seg_rgb.load_state_dict(best_state["seg_rgb"])
        recon_depth.load_state_dict(best_state["recon_depth"])
        seg_depth.load_state_dict(best_state["seg_depth"])

    reduced = evaluate_draem_rgbd(
        recon_rgb,
        seg_rgb,
        recon_depth,
        seg_depth,
        test_loader,
        device=device,
        amp_dtype=amp_dtype,
        fusion_rule=args.fusion_rule,
        selected_source=args.score_source,
    )
    if config.save_ckpt and best_state is not None:
        ckpt_dir = config.output_dir()
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, ckpt_dir / "best.pt")
    return synthesize_multisource_result(
        config,
        datasets,
        reduced,
        args,
        extra={
            "epochs": int(epochs),
            "lr": float(args.draem_lr),
            "best_val_loss": None if best_val == math.inf else float(best_val),
            "val_history": val_history,
            "synthetic_anomaly": "shared_perlin_self_cutpaste",
        },
    )


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
        input_mode="rgbd",
        depth_norm=args.depth_norm,
    )


def run_single_method(method: str, args) -> dict:
    config = build_config(args, method)
    seed_everything(config.seed)
    datasets = build_protocol_datasets(config)
    if method == "patchcore":
        return evaluate_patchcore_rgbd(config, datasets, args)
    if method == "padim":
        return evaluate_padim_rgbd(config, datasets, args)
    if method == "rd4ad_i":
        return fit_teacher_student_rgbd(
            config,
            datasets,
            method_name=method,
            lr=args.rd4ad_lr,
            epochs=args.rd4ad_epochs,
            loss_mode="rd4ad",
            args=args,
        )
    if method == "stfpm":
        return fit_teacher_student_rgbd(
            config,
            datasets,
            method_name=method,
            lr=args.stfpm_lr,
            epochs=args.stfpm_epochs,
            loss_mode="stfpm",
            args=args,
        )
    if method == "fastflow":
        return fit_fastflow_rgbd(config, datasets, args)
    if method == "draem":
        return fit_draem_rgbd(config, datasets, args)
    raise ValueError(f"Unsupported method: {method}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified RGB+Depth baseline launcher")
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
    parser.add_argument("--depth_norm", type=str, default="zscore", choices=["zscore", "minmax", "log"])
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
    parser.add_argument("--score_source", type=str, default="fusion", choices=list(FUSION_SOURCES))
    parser.add_argument("--fusion_rule", type=str, default="sum", choices=["sum", "max_norm"])

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
        print(f"[RUN] method={method} view_id={args.view_id} input_mode=rgbd")
        print("=" * 72)
        payload = run_single_method(method, args)
        summary_rows.append(
            {
                "method": payload["method"],
                "view_id": payload["view_id"],
                "image_auroc": payload["image_auroc"],
                "pixel_auroc": payload["pixel_auroc"],
                "image_auroc_rgb": payload["image_auroc_by_source"].get("rgb"),
                "image_auroc_depth": payload["image_auroc_by_source"].get("depth"),
                "image_auroc_fusion": payload["image_auroc_by_source"].get("fusion"),
                "pixel_auroc_rgb": payload["pixel_auroc_by_source"].get("rgb"),
                "pixel_auroc_depth": payload["pixel_auroc_by_source"].get("depth"),
                "pixel_auroc_fusion": payload["pixel_auroc_by_source"].get("fusion"),
                "score_source_selected": payload["score_source_selected"],
                "fusion_rule": payload["fusion_rule"],
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
            summary_csv = PROJ_ROOT / "$out" / "baselines_rgbd_smoke" / f"cam{args.view_id}" / "summary.csv"
        else:
            summary_csv = PROJ_ROOT / "results" / "baselines_rgbd" / f"cam{args.view_id}" / "summary.csv"
        if args.summary_json:
            summary_json = Path(args.summary_json)
        elif args.smoke:
            summary_json = PROJ_ROOT / "$out" / "baselines_rgbd_smoke" / f"cam{args.view_id}" / "summary.json"
        else:
            summary_json = PROJ_ROOT / "results" / "baselines_rgbd" / f"cam{args.view_id}" / "summary.json"
        write_summary_files(summary_rows, summary_csv, summary_json)
        print(f"[DONE] summary_csv={summary_csv}")
        print(f"[DONE] summary_json={summary_json}")


if __name__ == "__main__":
    main()
