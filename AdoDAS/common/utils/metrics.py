from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    roc_auc_score,
)

# 二分类F1分数，平均每个类别的F1分数
def binary_f1(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> float:
    preds = (probs >= threshold).astype(int) # 将概率转为0/1预测
    scores = []
    for c in range(probs.shape[1]):
        scores.append(f1_score(labels[:, c], preds[:, c], zero_division=0.0))
    return float(np.mean(scores))

# 每个类别的F1分数列表
def per_class_f1(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> list[float]:
    preds = (probs >= threshold).astype(int)
    return [
        float(f1_score(labels[:, c], preds[:, c], zero_division=0.0))
        for c in range(probs.shape[1])
    ]

# 计算每个类别的AUC分数，并返回平均AUC分数
def macro_auroc(probs: np.ndarray, labels: np.ndarray) -> float:
    scores = []
    for c in range(probs.shape[1]):
        unique = np.unique(labels[:, c]) # 检查该类别是否有正负样本
        if len(unique) < 2: # 如果只有一个类别（全是正样本或全是负样本），AUC无法计算，默认分数为0.0
            scores.append(0.0)
        else:
            scores.append(float(roc_auc_score(labels[:, c], probs[:, c])))
    return float(np.mean(scores))

# QWK计算函数，计算两个标签序列之间的二次加权Kappa分数
def _quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 4) -> float:
    N = num_classes # 4个类别，标签范围为0-3
    # 构建权重矩阵，w[i, j]表示将标签i预测为j的权重，权重根据标签之间的距离平方计算，范围在0到1之间
    w = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            w[i, j] = (i - j) ** 2 / ((N - 1) ** 2) # 平方惩罚，标签越远惩罚越大

    hist_true = np.bincount(y_true, minlength=N).astype(np.float64) # 每个类别的真实样本数量
    hist_pred = np.bincount(y_pred, minlength=N).astype(np.float64) # 每个类别的预测样本数量
    n = len(y_true)

    # 构建混淆矩阵O，O[i, j]表示真实标签为i且预测标签为j的样本数量
    O = np.zeros((N, N), dtype=np.float64)
    for t, p in zip(y_true, y_pred):
        O[t, p] += 1

    # 构建期望矩阵E，E[i,j] = (真实为i的样本数 * 预测为j的样本数) / 总样本数
    E = np.outer(hist_true, hist_pred) / n

    num = np.sum(w * O) # 实际加权误差
    den = np.sum(w * E) # 期望加权误差
    if den == 0:
        return 1.0 # 如果期望误差为0，说明所有样本都被正确分类，Kappa分数为1.0
    return 1.0 - num / den 

# 所有任务的平均QWK分数，计算每个类别的QWK分数并返回平均值
def mean_qwk(preds: np.ndarray, labels: np.ndarray) -> float:
    scores = []
    for c in range(preds.shape[1]):
        scores.append(_quadratic_weighted_kappa(labels[:, c], preds[:, c]))
    return float(np.mean(scores))


def per_item_qwk(preds: np.ndarray, labels: np.ndarray) -> list[float]:
    return [
        _quadratic_weighted_kappa(labels[:, c], preds[:, c])
        for c in range(preds.shape[1])
    ]

# 平均绝对误差，计算每个类别的MAE分数并返回平均值
def mean_mae(preds: np.ndarray, labels: np.ndarray) -> float:
    scores = []
    for c in range(preds.shape[1]):
        scores.append(float(mean_absolute_error(labels[:, c], preds[:, c])))
    return float(np.mean(scores))
