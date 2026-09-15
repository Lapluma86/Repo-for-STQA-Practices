# -*- coding: utf-8 -*-
"""
工程指标：可训练参数量、显存占用、推理时延、误报率等。
学术指标（I-AUC / P-AUC / PRO）继续使用 eval/eval_utils.py 与 eval/metrics_utils.py。
"""

import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


def count_trainable_params(model: nn.Module) -> Tuple[int, int, float]:
    """返回 (总参数量, 可训练参数量, 可训练比例)。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio = trainable / total if total > 0 else 0.0
    return total, trainable, ratio


def measure_inference_latency(
    model: nn.Module,
    sample_input,
    device: torch.device,
    n_warmup: int = 20,
    n_run: int = 100,
) -> float:
    """单图前向耗时（毫秒/图）。sample_input 必须是 tuple。"""
    assert isinstance(sample_input, (tuple, list)), "sample_input 需为 tuple/list"
    model.eval()
    use_cuda = device.type == "cuda"
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(*sample_input)
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_run):
            _ = model(*sample_input)
        if use_cuda:
            torch.cuda.synchronize()
        elapsed = time.time() - t0
    return elapsed / n_run * 1000.0


def measure_peak_gpu_memory(device: torch.device) -> float:
    """读取并返回 device 上的峰值显存（MB）。调用前请先 reset_peak_memory_stats。"""
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / 1024 ** 2


def compute_fp_per_image(
    scores: np.ndarray, labels: np.ndarray, threshold: float
) -> float:
    """图级误报：在给定阈值下，正常图被误判为异常的比例。"""
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    normal_mask = labels == 0
    n_normal = int(normal_mask.sum())
    if n_normal == 0:
        return 0.0
    return float((scores[normal_mask] > threshold).sum()) / n_normal
