"""
ordinal_loss_enhanced.py — 增强的序数回归损失（CORN + 可微 QWK 辅助损失）

问题诊断：
    A2 任务（21 项 DASS 量表，每项 0~3 分）使用序数回归 BCE 损失。

    当前损失（a2_ordinal_loss）的问题：
    1. BCE 对所有错误等权重惩罚：预测 0 vs 真实 3（差3级）和
       预测 2 vs 真实 3（差1级）在 BCE 看来贡献相同的损失
    2. 评价指标是 QWK（Quadratic Weighted Kappa），它对"差距大"的错误
       惩罚更重 → 训练优化的目标（BCE）与评价指标（QWK）不对齐
    3. runner.py 调用 a2_ordinal_loss(..., use_corn=True, use_qwk=True)
       但这些参数在 heads.py 中没有实现 → 运行时报错

    QWK 的关键特性：
        QWK 衡量"偏差的严重程度"：
        - 预测 2, 真实 3 → 惩罚权重 (2-3)² = 1
        - 预测 0, 真实 3 → 惩罚权重 (0-3)² = 9  ← 9倍惩罚！
        所以"减少大错误"对 QWK 的收益远大于"减少小错误"。

改进方案：
    1. CORN Loss (Conditional Ordinal Regression)：
       将序数回归建模为条件概率链，保证概率单调性
       P(Y≥k) = P(Y≥1) × P(Y≥2|Y≥1) × ... × P(Y≥k|Y≥k-1)

    2. Differentiable QWK Loss：
       直接优化 QWK 的可微版本作为辅助损失
       → 确保"训练时惩罚大错误更多" = "评价时惩罚大错误更多"

    3. Combined = BCE + α×CORN + β×QWK_loss

预期提升：QWK +3~8%，MAE -0.5~1.0

实施优先级：P0（修复代码缺失 + 直接对齐评价指标）
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def corn_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_thresholds: int = 3,
) -> torch.Tensor:
    """
    CORN Loss (Conditional Ordinal Regression with Neural Networks)
    参考: Shi et al., "CORN: Conditional Ordinal Regression for Neural Networks", PR 2021

    核心思想：
        传统序数回归：P(Y≥k) 独立建模 → 可能出现 P(Y≥2) > P(Y≥1) 的矛盾
        CORN：建模条件概率 P(Y≥k | Y≥k-1) → 天然保证单调性

    概率链分解：
        P(Y≥1) = σ(logit_1)
        P(Y≥2) = P(Y≥1) × P(Y≥2|Y≥1)  = σ(logit_1) × σ(logit_2)
        P(Y≥3) = P(Y≥2) × P(Y≥3|Y≥2)  = σ(logit_1) × σ(logit_2) × σ(logit_3)

        天然保证 P(Y≥3) ≤ P(Y≥2) ≤ P(Y≥1)

    CORN 损失的训练方式：
        对阈值 k，只在 labels ≥ k-1 的样本上训练条件概率 P(Y≥k | Y≥k-1)
        直觉："只有通过了前一关的人，才需要判断是否通过下一关"

    参数:
        logits: (B, n_items, n_thresholds) — 条件 logit
        labels: (B, n_items) — 整数标签 0~3

    返回:
        loss: 标量
    """
    B, I, T = logits.shape
    total_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    count = 0

    for k in range(T):
        # 条件训练集：只选取 labels >= k 的样本
        # 即"已经通过前 k 个阈值"的样本
        if k == 0:
            mask = torch.ones(B, I, dtype=torch.bool, device=logits.device)
        else:
            mask = labels >= k  # (B, I)

        if mask.sum() == 0:
            continue

        # 目标：在条件集内，labels >= k+1 的样本为正
        target = (labels >= (k + 1)).float()  # (B, I)

        # 只在条件集上计算 BCE
        loss_k = F.binary_cross_entropy_with_logits(
            logits[:, :, k][mask],
            target[mask],
            reduction="mean",
        )
        total_loss = total_loss + loss_k
        count += 1

    return total_loss / max(count, 1)


def differentiable_qwk_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_classes: int = 4,
    n_thresholds: int = 3,
) -> torch.Tensor:
    """
    可微 QWK 损失：直接优化 QWK 指标的可微近似

    核心思想：
        QWK 基于混淆矩阵计算 → 构造"软"混淆矩阵（概率版本）。

    步骤：
        1. logits → sigmoid → 累积概率 → 各等级概率分布 P(Y=k)
        2. 构造权重矩阵 W[i][j] = (i-j)²/((n-1)²)
        3. 构造预测的"软"分布和真实的 one-hot 分布
        4. 计算可微 QWK = 1 - Σ(W × O) / Σ(W × E)
           其中 O 是观察到的（软）混淆矩阵，E 是期望混淆矩阵

    为什么直接优化 QWK？
        - 评价指标就是 QWK → 直接对齐 → 无 surrogate gap
        - QWK 对"大错误"惩罚更重 → 模型学会避免 "0 vs 3" 这种灾难性错误

    参数:
        logits: (B, n_items, n_thresholds) — 序数回归 logits
        labels: (B, n_items) — 整数标签 0~3
        n_classes: 等级数
        n_thresholds: 阈值数

    返回:
        loss: 标量，1 - soft_QWK（越大越差）
    """
    B, I, T = logits.shape

    # Step 1: logits → 各等级概率分布
    # 使用单调约束的概率计算
    s = torch.sigmoid(logits)      # (B, I, T) — 累积概率
    p1 = s[..., 0]                 # P(Y≥1)
    p2 = torch.min(s[..., 1], p1)  # P(Y≥2) ≤ P(Y≥1)
    p3 = torch.min(s[..., 2], p2)  # P(Y≥3) ≤ P(Y≥2)

    # 各等级概率：(B, I, 4)
    P0 = 1.0 - p1
    P1 = p1 - p2
    P2 = p2 - p3
    P3 = p3
    pred_probs = torch.stack([P0, P1, P2, P3], dim=-1)  # (B, I, 4)
    # clamp 防止数值问题
    pred_probs = pred_probs.clamp(min=1e-7)

    # Step 2: 真实标签的 one-hot 分布
    # labels: (B, I) → (B, I, 4) one-hot
    true_onehot = F.one_hot(labels.long(), num_classes=n_classes).float()  # (B, I, 4)

    # Step 3: 构造 QWK 权重矩阵
    # W[i][j] = (i-j)² / (n_classes-1)²
    idx = torch.arange(n_classes, device=logits.device, dtype=torch.float32)
    weight_matrix = (idx.unsqueeze(0) - idx.unsqueeze(1)) ** 2  # (4, 4)
    weight_matrix = weight_matrix / (n_classes - 1) ** 2

    # Step 4: 计算软混淆矩阵（在 batch × items 上汇总）
    # 将 (B, I) 展平为 (B*I,)
    pred_flat = pred_probs.reshape(-1, n_classes)     # (B*I, 4)
    true_flat = true_onehot.reshape(-1, n_classes)    # (B*I, 4)
    N = pred_flat.shape[0]

    # 观察混淆矩阵 O[i][j] = Σ_n true_n[i] × pred_n[j] / N
    O = torch.matmul(true_flat.T, pred_flat) / N  # (4, 4)

    # 期望混淆矩阵 E[i][j] = (Σ_n true_n[i]) × (Σ_n pred_n[j]) / N²
    hist_true = true_flat.sum(dim=0) / N           # (4,)
    hist_pred = pred_flat.sum(dim=0) / N           # (4,)
    E = torch.outer(hist_true, hist_pred)          # (4, 4)

    # Step 5: QWK = 1 - Σ(W × O) / Σ(W × E)
    numerator = (weight_matrix * O).sum()
    denominator = (weight_matrix * E).sum().clamp(min=1e-7)
    soft_qwk = 1.0 - numerator / denominator

    # 返回 1 - QWK 作为损失（QWK 越高越好，所以最小化 1-QWK）
    return 1.0 - soft_qwk


def a2_ordinal_loss_enhanced(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
    use_corn: bool = False,
    use_qwk: bool = False,
    qwk_weight: float = 0.3,
) -> torch.Tensor:
    """
    A2 增强序数回归损失（兼容 runner.py 调用接口）

    损失组成：
        base_loss = ordinal_BCE（原始损失，保证基线性能）
        + use_corn=True  → + CORN_loss（条件序数回归，保证单调性）
        + use_qwk=True   → + qwk_weight × QWK_loss（直接优化评价指标）

    当 use_corn=False 且 use_qwk=False 时，退化为原始 a2_ordinal_loss。

    参数:
        logits:          (B, n_items, n_thresholds) — 序数回归 logits
        labels:          (B, n_items) — 整数标签 0~3
        pos_weight:      阈值级别的正样本权重
        label_smoothing: 标签平滑
        use_corn:        是否使用 CORN 条件序数损失
        use_qwk:         是否使用可微 QWK 辅助损失
        qwk_weight:      QWK 损失的权重

    返回:
        loss: 标量
    """
    n_thresholds = logits.size(-1)

    # 基础损失：标准序数回归 BCE
    from common.models.heads import A2OrdinalHead
    targets = A2OrdinalHead.build_ordinal_targets(labels, n_thresholds=n_thresholds)

    if label_smoothing > 0.0:
        targets = targets * (1.0 - label_smoothing) + 0.5 * label_smoothing

    base_loss = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight
    )

    total_loss = base_loss

    # CORN 辅助损失：条件概率链
    if use_corn:
        total_loss = total_loss + corn_loss(logits, labels, n_thresholds=n_thresholds)

    # 可微 QWK 辅助损失
    if use_qwk:
        qwk_loss = differentiable_qwk_loss(logits, labels, n_thresholds=n_thresholds)
        total_loss = total_loss + qwk_weight * qwk_loss

    return total_loss


# ============================================================
# 集成方式：替换 heads.py 中的 a2_ordinal_loss
# ============================================================
#
# 在 heads.py 中：
#
# from docs.optimize.loss_functions.ordinal_loss_enhanced import a2_ordinal_loss_enhanced
#
# # 替换
# a2_ordinal_loss = a2_ordinal_loss_enhanced
#
# 或者直接将本文件的函数复制到 heads.py 中。
