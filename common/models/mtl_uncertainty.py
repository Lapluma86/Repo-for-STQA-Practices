"""
不确定性加权多任务学习 (Uncertainty Weighting for Multi-Task Learning)

参考: Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics", CVPR 2018

核心思想：
    每个任务的损失权重不是固定的，而是通过可学习的"不确定性"参数自动调整。

    传统 MTL：L_total = w1×L1 + w2×L2 + w3×L3  (w1,w2,w3 需要手动调参)
    不确定性加权：L_total = (1/2σ1²)×L1 + log(σ1) + (1/2σ2²)×L2 + log(σ2) + ...

    σ 是可学习参数（任务的"不确定性"）：
        - σ 大 → 任务不确定性高 → 降低该任务的权重
        - σ 小 → 任务确定性高 → 提高该任务的权重
        - log(σ) 项防止 σ 趋向无穷大（正则化）

优势：
    1. 自动平衡多任务，无需手动调参
    2. 训练过程中动态调整权重
    3. 理论基础：贝叶斯深度学习中的同方差不确定性（homoscedastic uncertainty）
"""
from __future__ import annotations

import torch
import torch.nn as nn


class UncertaintyWeightedLoss(nn.Module):
    """
    不确定性加权多任务损失

    参数:
        n_tasks: 任务数量
        init_log_var: 初始 log(σ²) 值（默认0，即 σ²=1）
    """
    def __init__(self, n_tasks: int, init_log_var: float = 0.0, log_var_clamp: float | None = None):
        super().__init__()
        self.n_tasks = n_tasks
        self.log_var_clamp = log_var_clamp
        # 可学习参数：log(σ²)，用 log 保证 σ² > 0
        self.log_vars = nn.Parameter(torch.full((n_tasks,), init_log_var, dtype=torch.float32))

    def forward(self, losses: list[torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        """
        参数:
            losses: 各任务的损失列表 [L1, L2, ..., Ln]

        返回:
            total_loss: 加权后的总损失
            weights: 各任务的有效权重（用于日志记录）
        """
        if len(losses) != self.n_tasks:
            raise ValueError(f"Expected {self.n_tasks} losses, got {len(losses)}")

        total_loss = 0.0
        weights = {}

        for i, loss in enumerate(losses):
            log_var = self.log_vars[i]
            if self.log_var_clamp is not None:
                log_var = torch.clamp(log_var, -self.log_var_clamp, self.log_var_clamp)
            # 不确定性加权公式：(1 / 2σ²) × L + log(σ)
            # = (1 / 2exp(log_var)) × L + 0.5 × log_var
            precision = torch.exp(-log_var)  # 1/σ²
            weighted_loss = 0.5 * precision * loss + 0.5 * log_var
            total_loss = total_loss + weighted_loss

            # 记录有效权重（用于监控）
            weights[f"task_{i}_weight"] = precision.item()
            weights[f"task_{i}_sigma"] = torch.exp(0.5 * log_var).item()

        return total_loss, weights


class MultiTaskHead(nn.Module):
    """多任务预测头 — 仅保留 emotion_dims 作为弱正则化项"""

    def __init__(self, d_in: int, task_type: str, enable_emotion_dims: bool = True):
        super().__init__()
        self.task_type = task_type
        self.enable_emotion_dims = enable_emotion_dims

        if enable_emotion_dims:
            self.emotion_dim_head = nn.Sequential(
                nn.Linear(d_in, 64),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(64, 2),
            )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = {}
        if self.enable_emotion_dims:
            outputs["emotion_dims"] = self.emotion_dim_head(x)
        return outputs


def compute_auxiliary_losses(
    aux_outputs: dict[str, torch.Tensor],
    aux_targets: dict[str, torch.Tensor] | None,
) -> dict[str, torch.Tensor]:
    """计算辅助任务损失 — 仅 emotion_dims (弱正则化 MSE)"""
    losses = {}

    if aux_targets is None:
        for key in aux_outputs:
            losses[key] = aux_outputs[key].new_zeros(())
        return losses

    if "emotion_dims" in aux_outputs and "emotion_dims" in aux_targets:
        pred = torch.tanh(aux_outputs["emotion_dims"])
        target = aux_targets["emotion_dims"]
        mask = ~torch.isnan(target).any(dim=-1)
        if mask.any():
            losses["emotion_dims"] = nn.functional.mse_loss(pred[mask], target[mask])
        else:
            losses["emotion_dims"] = pred.new_zeros(())

    return losses
