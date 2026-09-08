# -*- coding: utf-8 -*-
"""Feature distribution and feature-stat helpers for rail PEFT/diagnostics."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Iterable, List, Tuple

import torch
import torch.nn.functional as F


@torch.no_grad()
def compute_depth_feature_stats(
    teacher_depth,
    dataloader,
    device: torch.device | str = "cuda",
    amp_context_factory=None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Compute per-layer channel mean/variance for teacher depth features."""
    teacher_depth.eval()
    device = torch.device(device)

    sums: List[torch.Tensor] = []
    sumsqs: List[torch.Tensor] = []
    counts: List[int] = []

    for batch_idx, data in enumerate(dataloader):
        depth = data["depth"].to(device, non_blocking=True)
        amp_ctx = amp_context_factory() if amp_context_factory is not None else nullcontext()
        with amp_ctx:
            feats = teacher_depth(depth)

        if batch_idx == 0:
            for feat in feats:
                channels = feat.shape[1]
                sums.append(torch.zeros(channels, device=device, dtype=torch.float64))
                sumsqs.append(torch.zeros(channels, device=device, dtype=torch.float64))
                counts.append(0)

        for layer_idx, feat in enumerate(feats):
            feat64 = feat.detach().to(dtype=torch.float64)
            sums[layer_idx] += feat64.sum(dim=(0, 2, 3))
            sumsqs[layer_idx] += (feat64 * feat64).sum(dim=(0, 2, 3))
            counts[layer_idx] += feat.shape[0] * feat.shape[2] * feat.shape[3]

    if not sums:
        raise RuntimeError("Cannot compute feature stats from an empty dataloader.")

    mu_list = [(s / n).to(dtype=torch.float32) for s, n in zip(sums, counts)]
    var_list = [
        (sq / n - mu.to(dtype=torch.float64) ** 2).clamp_min(1e-8).to(dtype=torch.float32)
        for sq, n, mu in zip(sumsqs, counts, mu_list)
    ]
    return mu_list, var_list


@torch.no_grad()
def compute_teacher_feature_means(
    teacher,
    dataloader,
    input_key: str,
    device: torch.device | str = "cuda",
    amp_context_factory=None,
    channels_last: bool = False,
) -> List[torch.Tensor]:
    """Compute per-layer teacher feature means and keep them as [1, C, 1, 1]."""
    teacher.eval()
    device = torch.device(device)

    sums: List[torch.Tensor] = []
    counts: List[int] = []

    for batch_idx, data in enumerate(dataloader):
        model_input = data[input_key].to(device, non_blocking=True)
        if channels_last and device.type == "cuda":
            model_input = model_input.contiguous(memory_format=torch.channels_last)
        amp_ctx = amp_context_factory() if amp_context_factory is not None else nullcontext()
        with amp_ctx:
            feats = teacher(model_input)

        if batch_idx == 0:
            for feat in feats:
                channels = feat.shape[1]
                sums.append(torch.zeros((1, channels, 1, 1), device=device, dtype=torch.float64))
                counts.append(0)

        for layer_idx, feat in enumerate(feats):
            feat64 = feat.detach().to(dtype=torch.float64)
            sums[layer_idx] += feat64.sum(dim=(0, 2, 3), keepdim=True)
            counts[layer_idx] += feat.shape[0] * feat.shape[2] * feat.shape[3]

    if not sums:
        raise RuntimeError("Cannot compute teacher feature means from an empty dataloader.")

    return [(s / n).to(dtype=torch.float32) for s, n in zip(sums, counts)]


def expand_feature_means(
    feature_means: Iterable[torch.Tensor],
    like_feats: Iterable[torch.Tensor],
) -> List[torch.Tensor]:
    """Broadcast [1, C, 1, 1] means to match teacher feature shapes."""
    expanded = []
    for mean_feat, like_feat in zip(feature_means, like_feats):
        mean_feat = mean_feat.to(device=like_feat.device, dtype=like_feat.dtype)
        expanded.append(
            mean_feat.expand(like_feat.shape[0], -1, like_feat.shape[2], like_feat.shape[3]).contiguous()
        )
    return expanded


def zeros_like_feature_list(like_feats: Iterable[torch.Tensor]) -> List[torch.Tensor]:
    """Create a zero-filled feature list with the same shapes as the reference list."""
    return [torch.zeros_like(feat) for feat in like_feats]


def fdm_loss(
    feats: Iterable[torch.Tensor],
    mu_ref: Iterable[torch.Tensor],
    var_ref: Iterable[torch.Tensor],
) -> torch.Tensor:
    """Feature Distribution Matching loss for channel mean/variance alignment."""
    loss = None
    for feat, mu, var in zip(feats, mu_ref, var_ref):
        feat32 = feat.float()
        cur_mu = feat32.mean(dim=(0, 2, 3))
        cur_var = feat32.var(dim=(0, 2, 3), unbiased=False).clamp_min(1e-8)
        layer_loss = F.mse_loss(cur_mu, mu.to(feat32.device)) + F.mse_loss(
            cur_var, var.to(feat32.device)
        )
        loss = layer_loss if loss is None else loss + layer_loss
    if loss is None:
        raise RuntimeError("fdm_loss received no feature tensors.")
    return loss
