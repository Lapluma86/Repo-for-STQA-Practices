"""
本模块定义了两个任务预测头和对应的损失函数。

架构定位：
    MTCNBackbone → (B, d_shared)
                        ├── A1Head        → (B, 3)      抑郁/焦虑/压力 logit
                        └── A2OrdinalHead → (B, 21, 3)  21项 × 3个累积阈值 logit

任务说明：
    A1：多标签二元分类，每个指标独立预测是否超过临床阈值（BCE损失）
    A2：序数回归，将整数等级（0-3）转化为3个累积二元问题（序数BCE损失）
        序数回归的优势：利用等级间的顺序关系，比普通4分类更稳健
        例：真实标签=2 → 目标向量=[1,1,0]（>=1通过, >=2通过, >=3不通过）
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class A1Head(nn.Module):
    """
    A1任务预测头 - 三分类二元分类

    用于预测三个心理健康指标: 抑郁(D)、焦虑(A)、压力(S)
    每个指标独立预测，输出一个概率值

    参数:
        d_in: 输入特征维度
        bias_init: 初始偏置值列表 [bias_D, bias_A, bias_S]
                   建议使用 log(p/(1-p)) 初始化，其中p是训练集正样本率
                   作用：让模型初始输出与数据分布一致，加速收敛

    示例:
        # 假设训练集中 D=15%, A=20%, S=18% 正样本
        bias_init = [log(0.15/0.85), log(0.20/0.80), log(0.18/0.82)]
        head = A1Head(d_in=256, bias_init=bias_init)
    """

    def __init__(self, d_in: int, bias_init: list[float] | None = None) -> None:
        super().__init__()
        # 单层线性：共享表示 → 3个独立logit（每个心理健康指标一个）
        self.fc = nn.Linear(d_in, 3)

        # 用先验正样本率初始化偏置：sigmoid(bias) = p_positive
        # 避免训练初期模型输出 0.5 而真实分布是 0.15，减少早期无效迭代
        if bias_init is not None:
            with torch.no_grad():
                self.fc.bias.copy_(torch.tensor(bias_init, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: 输入特征 (B, d_in)
        返回:
            logits: 未经过sigmoid的预测值 (B, 3)，对应 [logit_D, logit_A, logit_S]
        注意：返回logit而非概率，因为 binary_cross_entropy_with_logits 内部
              用数值稳定方式合并 sigmoid+BCE，避免极端值精度损失
        """
        return self.fc(x)

    @staticmethod
    def predict_probs(logits: torch.Tensor) -> torch.Tensor:
        """推理时将logits转换为概率值 (B, 3)，每个值在[0,1]"""
        return torch.sigmoid(logits)


class A2OrdinalHead(nn.Module):
    """
    A2任务预测头 - 序数回归

    预测21个心理评估项目的分数（0/1/2/3四个等级）。

    序数回归原理：
        将"预测整数k"转化为"预测k个累积二元问题"
        分数k → 目标向量 = [score>=1, score>=2, score>=3]
            k=0 → [0, 0, 0]
            k=1 → [1, 0, 0]
            k=2 → [1, 1, 0]
            k=3 → [1, 1, 1]
        三个概率满足单调性：p1 >= p2 >= p3

    参数:
        d_in:         输入特征维度
        n_items:      评估项目数（默认21）
        n_thresholds: 阈值数 = 等级数-1（默认3，对应0-3共4个等级）
    """

    def __init__(self, d_in: int, n_items: int = 21, n_thresholds: int = 3) -> None:
        super().__init__()
        self.n_items = n_items
        self.n_thresholds = n_thresholds
        # 输出 21×3=63 个logit，reshape后每项3个累积阈值logit
        self.fc = nn.Linear(d_in, n_items * n_thresholds)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        # (B, 63) → (B, 21, 3)：第i项的第j个阈值logit
        return self.fc(x).view(B, self.n_items, self.n_thresholds)

    @staticmethod
    def predict_int(logits: torch.Tensor) -> torch.Tensor:
        """
        简单解码：各阈值独立判断后求和
        缺陷：可能产生非单调结果（如[0,1,0]，跳过阈值2却超过阈值3）
        """
        return (torch.sigmoid(logits) > 0.5).long().sum(dim=-1)

    @staticmethod
    def predict_int_monotonic(logits: torch.Tensor) -> torch.Tensor:
        """
        单调解码（推荐）：强制累积概率满足 p1 >= p2 >= p3

        原理（累积链接模型）：
            p1 = sigmoid(logit_1)
            p2 = min(sigmoid(logit_2), p1)   # 强制单调
            p3 = min(sigmoid(logit_3), p2)

            P(score=0) = 1 - p1
            P(score=1) = p1 - p2
            P(score=2) = p2 - p3
            P(score=3) = p3
            # 四个概率之和=1，满足概率分布归一性

        返回概率最大的等级（argmax）
        """
        s = torch.sigmoid(logits)

        p1 = s[..., 0]
        p2 = torch.min(s[..., 1], p1)   # 强制 p2 <= p1
        p3 = torch.min(s[..., 2], p2)   # 强制 p3 <= p2

        P0 = 1.0 - p1
        P1 = p1 - p2
        P2 = p2 - p3
        P3 = p3
        class_probs = torch.stack([P0, P1, P2, P3], dim=-1)
        return class_probs.argmax(dim=-1)

    @staticmethod
    def predict_expectation(logits: torch.Tensor) -> torch.Tensor:
        """
        期望解码：计算等级期望值后取整

        E[score] = 0×P0 + 1×P1 + 2×P2 + 3×P3 = p1 + p2 + p3
        （三个累积概率之和等于期望值，是序数回归的数学性质）
        """
        s = torch.sigmoid(logits)
        p1 = s[..., 0]
        p2 = torch.min(s[..., 1], p1)
        p3 = torch.min(s[..., 2], p2)
        E = p1 + p2 + p3   # E[score] = p1 + p2 + p3
        return E.round().long().clamp(0, 3)

    @staticmethod
    def build_ordinal_targets(labels: torch.Tensor, n_thresholds: int = 3) -> torch.Tensor:
        """
        将整数标签转为序数目标向量（向量化实现）

        例：label=2, thresholds=[1,2,3]
            2>=1 → 1, 2>=2 → 1, 2>=3 → 0  →  [1,1,0]

        参数:
            labels: (B, n_items) 整数标签
        返回:
            targets: (B, n_items, n_thresholds) 二元目标，供BCE损失使用
        """
        B, I = labels.shape
        thresholds = torch.arange(1, n_thresholds + 1, device=labels.device).float()
        # labels.unsqueeze(-1): (B,I,1)，thresholds.view(1,1,-1): (1,1,T)，广播比较
        targets = (labels.unsqueeze(-1).float() >= thresholds.view(1, 1, -1)).float()
        return targets


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

    原理：对正/负样本使用不同的 focusing 参数
        L_pos = -(1 - p)^γ+ × log(p)         # γ+=0 → 不抑制正样本
        L_neg = -(p_clip)^γ- × log(1 - p_clip)  # γ-=2 → 强力抑制容易的负样本

    参数:
        logits:          (B, C) 未经 sigmoid 的预测值
        targets:         (B, C) 二元标签 {0, 1}
        gamma_neg:       负样本的 focusing 参数（越大越抑制简单负样本）
        gamma_pos:       正样本的 focusing 参数（通常为 0，不抑制）
        clip:            负样本概率截断阈值
        pos_weight:      正样本权重（可选）
        label_smoothing: 标签平滑系数
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

    原理：硬 F1 = 2TP / (2TP + FP + FN) 不可导，
    Soft-F1 用 sigmoid 概率替代硬判断，使 F1 变为连续可导函数。
    """
    probs = torch.sigmoid(logits)
    targets_f = targets.float()

    # 按类别计算 soft_TP, soft_FP, soft_FN
    soft_tp = (probs * targets_f).sum(dim=0)         # (C,)
    soft_fp = (probs * (1.0 - targets_f)).sum(dim=0)  # (C,)
    soft_fn = ((1.0 - probs) * targets_f).sum(dim=0)  # (C,)

    # Soft-F1 per class
    eps = 1e-7
    soft_f1 = (2 * soft_tp) / (2 * soft_tp + soft_fp + soft_fn + eps)  # (C,)

    # Macro-averaged（各类别等权重平均）
    return 1.0 - soft_f1.mean()


def a1_loss(
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
    A1 损失函数（增强版，兼容 runner.py 的调用接口）

    当 use_combined=False 时，退化为原始 BCE 损失（向后兼容）。
    当 use_combined=True 时，使用 ASL + Soft-F1 联合损失。

    参数:
        logits:          (B, C) 预测 logits
        targets:         (B, C) 二元标签
        pos_weight:      正样本权重
        label_smoothing: 标签平滑
        use_combined:    是否使用联合损失
        gamma_neg:       ASL 负样本 focusing 参数
        gamma_pos:       ASL 正样本 focusing 参数
        clip:            ASL 概率截断阈值
        soft_f1_weight:  Soft-F1 损失权重
    """
    if use_combined:
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

    # 原始 BCE 损失（向后兼容）
    if label_smoothing > 0.0:
        targets = targets.float() * (1.0 - label_smoothing) + 0.5 * label_smoothing
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)


def corn_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_thresholds: int = 3,
) -> torch.Tensor:
    """
    CORN Loss (Conditional Ordinal Regression with Neural Networks)
    参考: Shi et al., "CORN: Conditional Ordinal Regression for Neural Networks", PR 2021

    核心思想：建模条件概率 P(Y≥k | Y≥k-1) 而不是 P(Y≥k)，
    天然保证单调性 P(Y≥1) ≥ P(Y≥2) ≥ P(Y≥3)

    参数:
        logits: (B, n_items, n_thresholds) — 条件 logit
        labels: (B, n_items) — 整数标签 0~3
        n_thresholds: 阈值数

    返回:
        loss: 标量
    """
    B, I, T = logits.shape
    total_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    count = 0

    for k in range(T):
        # 条件训练集：只选取 labels >= k 的样本
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

    核心思想：构造"软"混淆矩阵（概率版本），直接优化 QWK 指标。
    QWK 对"大错误"的惩罚更重 (i-j)² → 避免灾难性错误 (0 vs 3)

    参数:
        logits: (B, n_items, n_thresholds) — 序数回归 logits
        labels: (B, n_items) — 整数标签 0~3
        n_classes: 等级数（通常为 4）
        n_thresholds: 阈值数（通常为 3）

    返回:
        loss: 标量，1 - soft_QWK
    """
    B, I, T = logits.shape

    # Step 1: logits → 各等级概率分布，使用单调约束
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
    pred_probs = pred_probs.clamp(min=1e-7)

    # Step 2: 真实标签的 one-hot 分布
    true_onehot = F.one_hot(labels.long(), num_classes=n_classes).float()  # (B, I, 4)

    # Step 3: 构造 QWK 权重矩阵 W[i][j] = (i-j)² / (n_classes-1)²
    idx = torch.arange(n_classes, device=logits.device, dtype=torch.float32)
    weight_matrix = (idx.unsqueeze(0) - idx.unsqueeze(1)) ** 2  # (4, 4)
    weight_matrix = weight_matrix / (n_classes - 1) ** 2

    # Step 4: 计算软混淆矩阵（在 batch × items 上汇总）
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

    # 返回 1 - QWK 作为损失
    return 1.0 - soft_qwk


def a2_ordinal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
    use_corn: bool = False,
    use_qwk: bool = False,
    qwk_weight: float = 0.3,
    loss_components: dict | None = None,
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
        if loss_components is not None:
            loss_components["qwk_loss"] = qwk_loss.item()

    return total_loss


# ---------------------------------------------------------------------------
# 辅助属性预测头（LUPI Phase 1：多任务辅助监督）
# ---------------------------------------------------------------------------

_AUX_ATTR_SPEC = {
    "aux_family":     {"num_classes": 6, "label_offset": -1},
    "aux_only_child": {"num_classes": 2, "label_offset":  0},
    "aux_favoritism": {"num_classes": 3, "label_offset": -1},
    "aux_academic":   {"num_classes": 3, "label_offset": -1},
    "aux_emotional":  {"num_classes": 3, "label_offset": -1},
}
_AUX_ATTR_NAMES = list(_AUX_ATTR_SPEC.keys())


class AuxAttributeHeads(nn.Module):
    """五个辅助属性的并行分类预测头。

    从 participant_repr 预测辅助属性，提供额外训练监督（LUPI）。
    迫使共享表示编码与心理状态相关的潜在特征。
    """

    def __init__(self, d_in: int, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(d_in, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, spec["num_classes"]),
            )
            for name, spec in _AUX_ATTR_SPEC.items()
        })

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {name: head(x) for name, head in self.heads.items()}


def aux_attribute_loss(
    aux_logits: dict[str, torch.Tensor],
    aux_attrs: torch.Tensor,
    weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """计算辅助属性分类损失，缺失值（-1）自动跳过。

    返回:
        total_loss: 加权总损失
        acc_dict:   各属性有效样本准确率
    """
    if weights is None:
        weights = {name: 0.05 for name in _AUX_ATTR_NAMES}

    total = torch.tensor(0.0, device=aux_attrs.device, dtype=torch.float32)
    acc_dict: dict[str, float] = {}

    for i, name in enumerate(_AUX_ATTR_NAMES):
        spec = _AUX_ATTR_SPEC[name]
        raw_vals = aux_attrs[:, i].long()
        valid = raw_vals >= 0
        if valid.sum() == 0:
            acc_dict[name] = -1.0
            continue

        labels = (raw_vals + spec["label_offset"]).clamp(min=0)
        logits = aux_logits[name][valid]
        targets = labels[valid]

        ce = F.cross_entropy(logits, targets)
        total = total + weights.get(name, 0.05) * ce

        preds = logits.argmax(dim=-1)
        acc_dict[name] = (preds == targets).float().mean().item()

    return total, acc_dict
