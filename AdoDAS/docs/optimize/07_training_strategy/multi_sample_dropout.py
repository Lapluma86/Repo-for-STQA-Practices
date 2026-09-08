"""
multi_sample_dropout.py — 多采样 Dropout 策略

问题诊断：
    当前模型在 fusion_mlp 和 task_head 中使用单次 Dropout。
    单次 Dropout 的问题：
    - 每个 batch 只看到一种 dropout mask → 梯度估计方差大
    - 需要更多的训练步数才能覆盖不同的 mask 组合
    - 对于小数据集（~1000 样本），训练步数有限，单次 Dropout 的正则化效果不充分

改进方案 — Multi-Sample Dropout：
    在前向传播中，对同一个特征用 K 个不同的 dropout mask 分别推理，
    得到 K 个预测结果，平均它们的损失。

    原理：
    - 等效于 K 倍的 batch size（梯度估计方差 ÷ K）
    - 等效于隐式 ensemble（K 个子网络的平均）
    - 参数量不变，只增加少量计算开销（task_head 很轻量）

    参考：Inoue, "Multi-Sample Dropout for Accelerated Training and Better Generalization", 2019

    K=5 时，实测在 NLP 分类任务上加速收敛 30% 并提升 1~2%。

预期提升：QWK/F1 +1~2%，训练收敛更快

实施优先级：P1
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MultiSampleDropoutHead(nn.Module):
    """
    多采样 Dropout 包装器：对任意 task_head 做多次 Dropout 采样

    使用方式：
        原始：logits = task_head(x)
        替换：logits = MultiSampleDropoutHead(task_head, K=5, dropout=0.3)(x)

    训练时：K 次 dropout → K 个 logits → 平均
    推理时：关闭 dropout → 等效于单次前向传播（与原始行为一致）

    参数:
        head:       任意 task_head（A1Head / A2OrdinalHead / CORALHead）
        n_samples:  dropout 采样次数（推荐 5~8）
        dropout:    dropout 比率（建议比原始值略高，如 0.3~0.5）
    """

    def __init__(
        self,
        head: nn.Module,
        n_samples: int = 5,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.head = head
        self.n_samples = n_samples
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_samples)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: (B, D) — 输入特征

        返回:
            logits: (B, ...) — 平均后的 logits
        """
        if not self.training:
            # 推理时不做多采样，直接前向传播
            return self.head(x)

        # 训练时：K 次独立 dropout → K 个 logits → 平均
        outputs = []
        for drop in self.dropouts:
            x_dropped = drop(x)
            logit = self.head(x_dropped)
            outputs.append(logit)

        # 逐元素平均
        return torch.stack(outputs).mean(dim=0)


class MultiSampleDropoutLoss(nn.Module):
    """
    多采样 Dropout 损失包装器

    与 MultiSampleDropoutHead 不同，这个版本在损失层面做平均，
    允许不同 dropout mask 对应的损失函数有不同的梯度信号。

    理论上比 logit 平均更好：
    - logit 平均：先平均再算损失 → 只有一个损失值
    - loss 平均：先算 K 个损失再平均 → K 个梯度信号 → 梯度估计更准确

    参数:
        head:       task_head
        loss_fn:    损失函数 (logits, labels) -> scalar
        n_samples:  采样次数
        dropout:    dropout 比率
    """

    def __init__(
        self,
        head: nn.Module,
        loss_fn: callable,
        n_samples: int = 5,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.head = head
        self.loss_fn = loss_fn
        self.n_samples = n_samples
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_samples)])

    def forward(self, x: torch.Tensor, labels: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        参数:
            x:      (B, D) — 输入特征
            labels: (B, ...) — 标签
            **kwargs: 传递给 loss_fn 的额外参数

        返回:
            loss: 标量 — K 个 dropout 版本损失的平均
        """
        if not self.training:
            logits = self.head(x)
            return self.loss_fn(logits, labels, **kwargs)

        total_loss = 0.0
        for drop in self.dropouts:
            x_dropped = drop(x)
            logits = self.head(x_dropped)
            loss = self.loss_fn(logits, labels, **kwargs)
            total_loss = total_loss + loss

        return total_loss / self.n_samples


# ============================================================
# 集成示例
# ============================================================
#
# # 方法 1：包装 task_head（简单）
# task_head_original = A1Head(d_shared, bias_init=bias_init)
# task_head = MultiSampleDropoutHead(task_head_original, n_samples=5, dropout=0.3)
# task_head = task_head.to(device)
#
# # 训练循环不需要修改
# logits = task_head(participant_repr)  # 训练时自动多采样
# loss = a1_loss(logits, targets)
#
# # 方法 2：包装损失函数（更优但需要小改训练循环）
# ms_loss = MultiSampleDropoutLoss(
#     head=A1Head(d_shared, bias_init=bias_init),
#     loss_fn=a1_loss,
#     n_samples=5,
#     dropout=0.3,
# ).to(device)
#
# loss = ms_loss(participant_repr, targets, pos_weight=pos_weight_t)
