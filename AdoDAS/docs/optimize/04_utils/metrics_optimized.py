"""
metrics_optimized.py — 优化后的评估指标模块

相比原始 metrics.py 的改进：
1. [P0] 新增 optimize_f1_thresholds：逐类搜索最优 F1 阈值
2. [P0] 新增 binary_f1_with_thresholds：用自定义阈值计算 F1
3. [P0] 新增 optimize_qwk_offsets：搜索 QWK 最优偏移量
4. [P2] _quadratic_weighted_kappa 向量化，大数据集加速 ~5x

可直接替换 common/utils/metrics.py，接口向后兼容。
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    roc_auc_score,
)


# ============================================================
# 原有函数（保持向后兼容）
# ============================================================

def binary_f1(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> float:
    """二分类 F1，所有类别用统一阈值，返回宏平均。"""
    preds = (probs >= threshold).astype(int)
    scores = []
    for c in range(probs.shape[1]):
        scores.append(f1_score(labels[:, c], preds[:, c], zero_division=0.0))
    return float(np.mean(scores))


def per_class_f1(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> list[float]:
    """返回每个类别的 F1 列表。"""
    preds = (probs >= threshold).astype(int)
    return [
        float(f1_score(labels[:, c], preds[:, c], zero_division=0.0))
        for c in range(probs.shape[1])
    ]


def macro_auroc(probs: np.ndarray, labels: np.ndarray) -> float:
    """宏平均 AUROC，跳过只有单一标签的类。"""
    scores = []
    for c in range(probs.shape[1]):
        unique = np.unique(labels[:, c])
        if len(unique) < 2:
            scores.append(0.0)
        else:
            scores.append(float(roc_auc_score(labels[:, c], probs[:, c])))
    return float(np.mean(scores))


def _quadratic_weighted_kappa(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 4
) -> float:
    """
    QWK 计算（向量化版本）。

    优化点：
    - 权重矩阵 w 用广播代替双重循环
    - 混淆矩阵 O 用 np.add.at 代替 Python 循环
    """
    N = num_classes

    # 向量化构建权重矩阵：w[i,j] = (i-j)^2 / (N-1)^2
    idx = np.arange(N, dtype=np.float64)
    w = (idx[:, None] - idx[None, :]) ** 2 / ((N - 1) ** 2)

    hist_true = np.bincount(y_true, minlength=N).astype(np.float64)
    hist_pred = np.bincount(y_pred, minlength=N).astype(np.float64)
    n = len(y_true)

    # 向量化构建混淆矩阵：避免 Python for 循环
    O = np.zeros((N, N), dtype=np.float64)
    np.add.at(O, (y_true, y_pred), 1)

    E = np.outer(hist_true, hist_pred) / n

    num = np.sum(w * O)
    den = np.sum(w * E)
    if den == 0:
        return 1.0
    return 1.0 - num / den


def mean_qwk(preds: np.ndarray, labels: np.ndarray) -> float:
    """所有任务的平均 QWK。"""
    scores = []
    for c in range(preds.shape[1]):
        scores.append(_quadratic_weighted_kappa(labels[:, c], preds[:, c]))
    return float(np.mean(scores))


def per_item_qwk(preds: np.ndarray, labels: np.ndarray) -> list[float]:
    """返回每个任务的 QWK 列表。"""
    return [
        _quadratic_weighted_kappa(labels[:, c], preds[:, c])
        for c in range(preds.shape[1])
    ]


def mean_mae(preds: np.ndarray, labels: np.ndarray) -> float:
    """平均 MAE。"""
    scores = []
    for c in range(preds.shape[1]):
        scores.append(float(mean_absolute_error(labels[:, c], preds[:, c])))
    return float(np.mean(scores))


# ============================================================
# 新增：阈值优化（P0）
# ============================================================

def optimize_f1_thresholds(
    probs: np.ndarray,
    labels: np.ndarray,
    search_range: tuple[float, float] = (0.15, 0.85),
    step: float = 0.01,
) -> list[float]:
    """
    逐类搜索最优 F1 阈值。

    原理：
    - 对每个二分类任务，遍历 [search_range[0], search_range[1]] 范围内的阈值
    - 选择使该类 F1 最大化的阈值
    - 不同类别的最优阈值通常不同（取决于正负样本比例和模型置信度分布）

    Args:
        probs: 模型输出概率, shape (n_samples, n_classes)
        labels: 真实标签, shape (n_samples, n_classes)
        search_range: 搜索范围，默认 [0.15, 0.85]
        step: 搜索步长，默认 0.01

    Returns:
        每个类别的最优阈值列表

    用法：
        # 在验证集上搜索
        thresholds = optimize_f1_thresholds(val_probs, val_labels)
        # 用搜索到的阈值在测试集上评估
        f1 = binary_f1_with_thresholds(test_probs, test_labels, thresholds)
    """
    thresholds = np.arange(search_range[0], search_range[1] + step, step)
    n_classes = probs.shape[1]
    best_thresholds = []

    for c in range(n_classes):
        best_t, best_f1 = 0.5, 0.0
        for t in thresholds:
            preds = (probs[:, c] >= t).astype(int)
            f = f1_score(labels[:, c], preds, zero_division=0.0)
            if f > best_f1:
                best_f1 = f
                best_t = float(t)
        best_thresholds.append(best_t)

    return best_thresholds


def binary_f1_with_thresholds(
    probs: np.ndarray,
    labels: np.ndarray,
    thresholds: list[float],
) -> float:
    """
    用逐类自定义阈值计算 F1 宏平均。

    与 binary_f1(threshold=0.5) 的区别：每个类别用独立的阈值，
    而非统一的 0.5。配合 optimize_f1_thresholds 使用。

    Args:
        probs: 模型输出概率, shape (n_samples, n_classes)
        labels: 真实标签, shape (n_samples, n_classes)
        thresholds: 每个类别的阈值列表, len = n_classes

    Returns:
        宏平均 F1 分数
    """
    scores = []
    for c, t in enumerate(thresholds):
        preds = (probs[:, c] >= t).astype(int)
        scores.append(f1_score(labels[:, c], preds, zero_division=0.0))
    return float(np.mean(scores))


def per_class_f1_with_thresholds(
    probs: np.ndarray,
    labels: np.ndarray,
    thresholds: list[float],
) -> list[float]:
    """用逐类自定义阈值计算每个类别的 F1。"""
    return [
        float(f1_score(labels[:, c], (probs[:, c] >= t).astype(int), zero_division=0.0))
        for c, t in enumerate(thresholds)
    ]


def optimize_qwk_offsets(
    raw_preds: np.ndarray,
    labels: np.ndarray,
    num_classes: int = 4,
    search_range: tuple[float, float] = (-1.5, 1.5),
    step: float = 0.1,
) -> list[float]:
    """
    搜索 QWK 最优偏移量。

    原理：
    - 对模型输出的连续预测值施加偏移量后再取整为离散类别
    - 不同任务的最优偏移量不同
    - 这是一种后处理校准，不需要重新训练模型

    Args:
        raw_preds: 模型原始连续预测值, shape (n_samples, n_tasks)
        labels: 真实整数标签, shape (n_samples, n_tasks)
        num_classes: 类别数，默认 4（标签 0-3）
        search_range: 偏移量搜索范围
        step: 搜索步长

    Returns:
        每个任务的最优偏移量列表

    用法：
        offsets = optimize_qwk_offsets(val_preds, val_labels)
        # 应用偏移量
        calibrated = np.clip(np.round(test_preds + offsets), 0, 3).astype(int)
        qwk = mean_qwk(calibrated, test_labels)
    """
    offset_candidates = np.arange(search_range[0], search_range[1] + step, step)
    n_tasks = raw_preds.shape[1]
    best_offsets = []

    for c in range(n_tasks):
        best_offset, best_qwk = 0.0, -1.0
        for offset in offset_candidates:
            # 施加偏移 → 取整 → 裁剪到有效范围
            adjusted = np.clip(
                np.round(raw_preds[:, c] + offset).astype(int),
                0, num_classes - 1
            )
            qwk = _quadratic_weighted_kappa(labels[:, c], adjusted, num_classes)
            if qwk > best_qwk:
                best_qwk = qwk
                best_offset = float(offset)
        best_offsets.append(best_offset)

    return best_offsets
