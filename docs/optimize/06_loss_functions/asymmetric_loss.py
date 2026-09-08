"""
asymmetric_loss.py — 非对称损失 (Asymmetric Loss, ASL) + Soft-F1 联合损失

问题诊断：
    A1 任务（抑郁/焦虑/压力二元分类）面临严重的类别不平衡：
    正样本率通常只有 15~25%。

    标准 BCE 的问题：
    - 负样本占多数 → 梯度被"简单负样本"主导 → 模型偏向预测"正常"
    - pos_weight 虽然放大正样本梯度，但同时放大了噪声标签的影响
    - 即使预测偏差很小（如 p=0.05 vs p=0.01），BCE 也会产生梯度
      → 大量"容易"的负样本消耗了训练容量

    当前代码的问题：
    runner.py 调用 a1_loss(..., use_combined=True, gamma_neg=2, soft_f1_weight=0.3)
    但 heads.py 中的 a1_loss 函数根本不支持这些参数 → 运行时会报错。

改进方案：
    1. ASL (Asymmetric Loss)：对正/负样本使用不同的 focusing 参数
       - gamma_neg=2：抑制容易的负样本（focal loss 思想）
       - gamma_pos=0：不抑制正样本（正样本本来就少，每个都重要）
       - clip：硬截断极低概率的负样本梯度

    2. Soft-F1 Loss：直接优化 F1 指标的可微版本
       - F1 = 2TP / (2TP + FP + FN)，离散不可导
       - Soft-F1 用概率替代硬分类，使 F1 变为连续可导函数
       - 解决了 "优化 BCE 不等于优化 F1" 的指标不对齐问题

    3. Combined Loss = α × ASL + β × Soft-F1
       → 兼顾样本级优化（ASL）和集合级优化（Soft-F1）

预期提升：A1 F1 +3~8%

实施优先级：P0（修复代码缺失 + 直接提升核心指标）
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def asymmetric_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma_neg: float = 2.0,
    gamma_pos: float = 0.0,
    clip: float = 0.05,
    pos_weight: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """
    非对称损失 (Asymmetric Loss for Multi-Label Classification)
    参考: Ridnik et al., "Asymmetric Loss For Multi-Label Classification", ICCV 2021

    原理：
        标准 Focal Loss 对正/负样本使用相同的 gamma → 抑制简单负样本的同时
        也抑制了稀有但确定的正样本。ASL 对两者用不同 gamma：

        L_pos = -(1 - p)^γ+ × log(p)         # γ+=0 → 不抑制正样本
        L_neg = -(p_clip)^γ- × log(1 - p_clip)  # γ-=2 → 强力抑制容易的负样本

        p_clip = max(p - clip, 0)：
            如果模型已经很确信是负样本（p < clip），直接将概率截断为 0，
            完全消除这些样本的梯度。比 focal loss 更激进地抑制简单负样本。

    参数:
        logits:          (B, C) 未经 sigmoid 的预测值
        targets:         (B, C) 二元标签 {0, 1}
        gamma_neg:       负样本的 focusing 参数（越大越抑制简单负样本）
        gamma_pos:       正样本的 focusing 参数（通常为 0，不抑制）
        clip:            负样本概率截断阈值
        pos_weight:      正样本权重（可选）
        label_smoothing: 标签平滑系数

    返回:
        loss: 标量
    """
    # 标签平滑
    if label_smoothing > 0.0:
        targets = targets.float() * (1.0 - label_smoothing) + 0.5 * label_smoothing

    # Sigmoid 概率
    probs = torch.sigmoid(logits)  # (B, C)

    # 正样本部分
    pos_probs = probs                             # P(y=1)
    # 负样本部分（带截断）
    neg_probs = (probs - clip).clamp(min=0)       # 截断后的概率

    # Focal 调制因子
    # 正样本：(1-p)^γ+ — γ+=0 时为 1，不调制
    pos_weight_factor = (1.0 - pos_probs) ** gamma_pos if gamma_pos > 0 else 1.0
    # 负样本：p^γ- — 概率越高（越"难"）权重越大，概率越低（越"易"）权重越小
    neg_weight_factor = neg_probs ** gamma_neg

    # 数值稳定的 log
    eps = 1e-7
    pos_log = torch.log(pos_probs.clamp(min=eps))
    neg_log = torch.log((1.0 - neg_probs).clamp(min=eps))

    # 分别计算正/负样本损失
    loss = -(targets * pos_weight_factor * pos_log +
             (1 - targets) * neg_weight_factor * neg_log)

    # 可选：正样本加权
    if pos_weight is not None:
        loss = loss * torch.where(targets > 0.5, pos_weight, torch.ones_like(pos_weight))

    return loss.mean()


def soft_f1_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Soft-F1 损失：直接优化 F1 指标的可微版本

    原理：
        硬 F1 = 2TP / (2TP + FP + FN)，需要离散的 0/1 预测 → 不可导
        Soft-F1 用 sigmoid 概率替代硬判断：
            soft_TP = Σ(p_i × y_i)             — 正确预测为正的"软"计数
            soft_FP = Σ(p_i × (1 - y_i))       — 错误预测为正的"软"计数
            soft_FN = Σ((1 - p_i) × y_i)       — 遗漏正样本的"软"计数
            soft_F1 = 2×soft_TP / (2×soft_TP + soft_FP + soft_FN)

        损失 = 1 - soft_F1（越大越差）

    为什么需要 Soft-F1？
        BCE 优化的是每个样本的独立概率，不关心全局的精度/召回权衡。
        Soft-F1 直接优化集合级别的 F1 指标，确保"优化目标 = 评价指标"。

    参数:
        logits:  (B, C) 未经 sigmoid 的预测值
        targets: (B, C) 二元标签

    返回:
        loss: 标量，1 - macro_soft_f1
    """
    probs = torch.sigmoid(logits)
    targets_f = targets.float()

    # 按类别计算 soft_TP, soft_FP, soft_FN
    # dim=0：在 batch 维度求和，得到每个类别的统计量
    soft_tp = (probs * targets_f).sum(dim=0)         # (C,)
    soft_fp = (probs * (1.0 - targets_f)).sum(dim=0)  # (C,)
    soft_fn = ((1.0 - probs) * targets_f).sum(dim=0)  # (C,)

    # Soft-F1 per class
    eps = 1e-7
    soft_f1 = (2 * soft_tp) / (2 * soft_tp + soft_fp + soft_fn + eps)  # (C,)

    # Macro-averaged（各类别等权重平均）
    return 1.0 - soft_f1.mean()


def combined_a1_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
    gamma_neg: float = 2.0,
    gamma_pos: float = 0.0,
    clip: float = 0.05,
    soft_f1_weight: float = 0.3,
) -> torch.Tensor:
    """
    A1 联合损失 = ASL + α × Soft-F1

    设计理由：
        - ASL 提供样本级的精细梯度信号，处理类别不平衡
        - Soft-F1 提供集合级的指标对齐，确保优化方向正确
        - 两者互补：ASL 防止模型懒惰（全预测负），Soft-F1 防止精度/召回失衡

    参数:
        logits:          (B, C) 预测 logits
        targets:         (B, C) 二元标签
        pos_weight:      正样本权重
        label_smoothing: 标签平滑
        gamma_neg:       ASL 负样本 focusing 参数
        gamma_pos:       ASL 正样本 focusing 参数
        clip:            ASL 概率截断阈值
        soft_f1_weight:  Soft-F1 损失权重

    返回:
        loss: 标量
    """
    asl = asymmetric_loss(
        logits, targets,
        gamma_neg=gamma_neg,
        gamma_pos=gamma_pos,
        clip=clip,
        pos_weight=pos_weight,
        label_smoothing=label_smoothing,
    )

    sf1 = soft_f1_loss(logits, targets)

    return asl + soft_f1_weight * sf1


def a1_loss_enhanced(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
    use_combined: bool = False,
    gamma_neg: float = 2.0,
    gamma_pos: float = 0.0,
    clip: float = 0.05,
    soft_f1_weight: float = 0.3,
) -> torch.Tensor:
    """
    A1 损失函数的增强版（兼容 runner.py 的调用接口）

    当 use_combined=False 时，退化为原始 BCE 损失（向后兼容）。
    当 use_combined=True 时，使用 ASL + Soft-F1 联合损失。

    这个函数可以直接替换 heads.py 中的 a1_loss 函数。
    """
    if use_combined:
        return combined_a1_loss(
            logits, targets,
            pos_weight=pos_weight,
            label_smoothing=label_smoothing,
            gamma_neg=gamma_neg,
            gamma_pos=gamma_pos,
            clip=clip,
            soft_f1_weight=soft_f1_weight,
        )

    # 原始 BCE 损失（向后兼容）
    if label_smoothing > 0.0:
        targets = targets.float() * (1.0 - label_smoothing) + 0.5 * label_smoothing
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)


# ============================================================
# 集成方式：替换 heads.py 中的 a1_loss 函数
# ============================================================
#
# 在 heads.py 中：
#
# from docs.optimize.loss_functions.asymmetric_loss import a1_loss_enhanced
#
# # 替换原来的 a1_loss 函数
# a1_loss = a1_loss_enhanced
#
# 或者直接将本文件的函数复制到 heads.py 中，替换原有的 a1_loss。
