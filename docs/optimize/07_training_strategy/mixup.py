"""
mixup.py — Mixup 数据增强（特征空间和标签空间的插值）

问题诊断：
    当前训练中仅有 feature_noise（加高斯噪声）和 session_drop（随机丢弃会话）
    两种正则化手段。对于 ~1000 样本的小数据集，这远远不够。

    过拟合的典型表现：
    - 训练 loss 持续下降，验证 loss 在 epoch 10 左右开始上升
    - 训练集 QWK > 0.9，验证集 QWK 只有 0.4~0.6
    - 模型对特定参与者的记忆比对通用模式的学习更强

改进方案 — Mixup：
    在特征空间对两个样本做线性插值，同时插值标签：
        x_mix = λ × x_i + (1-λ) × x_j
        y_mix = λ × y_i + (1-λ) × y_j

    其中 λ ~ Beta(α, α)，α 越小 → λ 越集中在 0 或 1（轻微混合），
    α 越大 → λ 越均匀（强混合）。

    Mixup 的理论基础：
    1. 扩展了训练分布的支撑集（support），等效于无穷多的数据增强
    2. 鼓励模型学习线性插值关系，作为正则化
    3. 软化决策边界，减少过拟合

    适用场景：
    - A1（二元分类）：标签本身就是 [0,1]，连续插值自然
    - A2（序数回归）：标签 0~3，插值后 y_mix ∈ [0,3]，
      序数回归目标 [y≥1, y≥2, y≥3] 也可以连续化

预期提升：QWK +2~5%（尤其在小数据集上效果显著）

实施优先级：P0
"""
from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np


def sample_mixup_lambda(alpha: float, batch_size: int, device: torch.device) -> torch.Tensor:
    """
    从 Beta 分布采样 Mixup 系数

    Beta(α, α) 的性质：
        α=0.2 → λ 集中在 0 和 1 附近（轻微混合，推荐起始值）
        α=0.5 → 中等混合
        α=1.0 → λ 均匀分布在 [0,1]（强混合）
        α=2.0 → λ 集中在 0.5 附近（几乎等量混合）

    使用 max(λ, 1-λ) 保证主样本权重 ≥ 0.5，避免标签翻转。

    返回:
        lam: (B, 1) — 每个样本独立采样一个 λ
    """
    if alpha <= 0:
        return torch.ones(batch_size, 1, device=device)

    lam = np.random.beta(alpha, alpha, size=(batch_size,))
    # 保证 λ ≥ 0.5（主样本占多数）
    lam = np.maximum(lam, 1 - lam)
    return torch.tensor(lam, dtype=torch.float32, device=device).unsqueeze(-1)


class ParticipantMixup(nn.Module):
    """
    参与者级 Mixup：在 participant_repr 层面做插值

    为什么在 participant_repr 而不是原始特征上做 Mixup？
        1. 原始特征包含多个模态、变长序列 → 结构复杂，直接插值不自然
        2. participant_repr 是固定大小向量 → 插值操作简单直接
        3. 在高层特征上混合 → 等效于更强的正则化
        4. 经验上，特征空间 Mixup（Manifold Mixup）比输入空间 Mixup 效果更好

    数据流：
        原始：participant_repr → task_head → loss(pred, label)
        Mixup：
            participant_repr_i, participant_repr_j = 随机配对
            repr_mix = λ × repr_i + (1-λ) × repr_j
            label_mix = λ × label_i + (1-λ) × label_j
            repr_mix → task_head → loss(pred, label_mix)

    参数:
        alpha: Beta 分布参数（推荐 0.2~0.4）
    """

    def __init__(self, alpha: float = 0.2):
        super().__init__()
        self.alpha = alpha

    def forward(
        self,
        repr: torch.Tensor,     # (B, D) — 参与者表示
        labels: torch.Tensor,   # (B, ...) — 标签（任意形状的后续维度）
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        返回:
            mixed_repr:   (B, D)   — 混合后的表示
            mixed_labels: (B, ...) — 混合后的标签
        """
        if not self.training or self.alpha <= 0:
            return repr, labels

        B = repr.size(0)
        device = repr.device

        # 采样 λ
        lam = sample_mixup_lambda(self.alpha, B, device)  # (B, 1)

        # 随机配对：打乱 batch 顺序
        perm = torch.randperm(B, device=device)

        # 特征插值
        mixed_repr = lam * repr + (1.0 - lam) * repr[perm]

        # 标签插值
        # labels 可能是 (B, 3) for A1 或 (B, 21) for A2
        # lam: (B, 1) 可以广播到 (B, *)
        lam_label = lam
        # 如果标签有额外维度，扩展 lam
        while lam_label.dim() < labels.dim():
            lam_label = lam_label.unsqueeze(-1)

        mixed_labels = lam_label * labels.float() + (1.0 - lam_label) * labels[perm].float()

        return mixed_repr, mixed_labels


class SessionMixup(nn.Module):
    """
    会话级 Mixup：在 session_reprs 层面做插值

    在 GroupedModel 内部，对同一参与者的不同会话表示做插值，
    相当于合成"虚拟会话"。

    这比参与者级 Mixup 更细粒度：
    - 参与者级：混合两个不同人的表示 → 合成虚拟参与者
    - 会话级：混合同一个人的不同会话 → 合成虚拟会话

    参数:
        alpha: Beta 分布参数
        intra_participant: 是否只在同一参与者内混合（True）
                           还是跨参与者混合（False）
    """

    def __init__(self, alpha: float = 0.2, intra_participant: bool = True):
        super().__init__()
        self.alpha = alpha
        self.intra_participant = intra_participant

    def forward(
        self,
        session_reprs: torch.Tensor,  # (B*4, D)
        n_participants: int,
    ) -> torch.Tensor:
        """
        参数:
            session_reprs: (B*4, D) — 所有会话的表示
            n_participants: B — 参与者数量

        返回:
            mixed: (B*4, D) — 混合后的会话表示
        """
        if not self.training or self.alpha <= 0:
            return session_reprs

        D = session_reprs.size(-1)
        B = n_participants

        if self.intra_participant:
            # 参与者内混合：(B*4, D) → (B, 4, D)
            grid = session_reprs.view(B, 4, D)

            # 对每个参与者的4个会话做随机配对
            perm = torch.stack([torch.randperm(4, device=session_reprs.device) for _ in range(B)])
            # (B, 4) — 打乱后的会话索引
            grid_perm = grid[torch.arange(B).unsqueeze(1), perm]

            lam = sample_mixup_lambda(self.alpha, B * 4, session_reprs.device).view(B, 4, 1)
            mixed = lam * grid + (1 - lam) * grid_perm
            return mixed.view(B * 4, D)
        else:
            # 跨参与者混合
            total = session_reprs.size(0)
            perm = torch.randperm(total, device=session_reprs.device)
            lam = sample_mixup_lambda(self.alpha, total, session_reprs.device)
            return lam * session_reprs + (1 - lam) * session_reprs[perm]


# ============================================================
# 集成示例
# ============================================================
#
# # 在训练循环中使用 ParticipantMixup：
# mixup = ParticipantMixup(alpha=0.2)
#
# for batch in train_loader:
#     ...
#     out = grouped_model(flat_batch, B, session_valid)
#     participant_repr = out["participant_repr"]
#
#     # Mixup 只在训练时生效
#     mixed_repr, mixed_labels = mixup(participant_repr, targets)
#
#     # 用混合后的表示和标签计算损失
#     p_logits = task_head(mixed_repr)
#     if task == "a1":
#         main_loss = a1_loss(p_logits, mixed_labels, ...)
#     else:
#         # A2 需要将连续标签适配到序数回归
#         main_loss = a2_ordinal_loss(p_logits, mixed_labels.round().long(), ...)
