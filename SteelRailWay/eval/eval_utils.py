# -*- coding: utf-8 -*-
"""
异常检测评估工具。

本模块提供：
    1. 由学生/教师特征计算 **异常热图** 的两种方式：
       - ``cal_anomaly_map`` : 基于余弦距离 (1 - cos_sim)
       - ``cal_l2dis``        : 基于 L2 距离
    2. 热图归一化和着色：``min_max_norm`` / ``cvt2heatmap``
    3. PRO（Per-Region Overlap）曲线下面积：
       - ``compute_pro``   : 截取 FPR ∈ [0, 0.3] 区间的 AUC
       - ``compute_pro_1`` : 截取 FPR ∈ [0, 0.01] 区间的 AUC（更严格）

参考实现：https://github.com/hq-deng/RD4AD/blob/main/test.py#L64
"""

import torch
import numpy as np
from torch.nn import functional as F
import cv2
from sklearn.metrics import auc
from skimage import measure
import pandas as pd
from numpy import ndarray
from statistics import mean


def cal_anomaly_map(fs_list, ft_list, out_size=256, amap_mode='mul'):
    """用余弦距离在多尺度上计算异常图，并融合为单张热图。

    参数：
        fs_list  : 学生网络（解码器）的多尺度特征列表 [B, C, H, W]
        ft_list  : 教师网络（编码器）的多尺度特征列表（逐层一一对应）
        out_size : 输出热图的空间分辨率（上采样到此大小后再融合）
                   可以是 int（正方形）或 tuple (H, W)
        amap_mode: 'mul' -> 各尺度逐元素相乘（强调共同响应）
                   其它   -> 逐元素相加（更鲁棒）
    返回：
        anomaly_map : 融合后的异常热图 [B, H, W] 或单张图时 [H, W]
        a_map_list  : 每个尺度的单独热图列表（便于可视化）
    """
    if isinstance(out_size, int):
        out_size_tuple = (out_size, out_size)
    else:
        out_size_tuple = out_size

    batch_size = fs_list[0].shape[0]

    # 在 GPU 上累积（延迟 CPU 传输，减少 CUDA 同步）
    if amap_mode == 'mul':
        anomaly_map = torch.ones((batch_size, *out_size_tuple), device=fs_list[0].device)
    else:
        anomaly_map = torch.zeros((batch_size, *out_size_tuple), device=fs_list[0].device)

    a_map_list = []
    for i in range(len(fs_list)):
        # AMP/bf16 前向会产生 BFloat16 特征，但后续需要转 NumPy。
        # NumPy 不支持 BFloat16，这里统一转 fp32，也让指标计算更稳定。
        fs = fs_list[i].float()
        ft = ft_list[i].float()
        a_map = 1 - F.cosine_similarity(fs, ft)  # [B, H, W]
        a_map = torch.unsqueeze(a_map, dim=1)    # [B, 1, H, W]
        a_map = F.interpolate(a_map, size=out_size_tuple, mode='bilinear', align_corners=True)
        a_map = a_map[:, 0, :, :]                # [B, H, W] on GPU

        # 保存在 GPU 上的副本（按需转 numpy）
        a_map_np = a_map.float().to('cpu').detach().numpy()
        a_map_list.append(a_map_np)

        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map

    # 最终结果一次性转 CPU
    anomaly_map = anomaly_map.float().to('cpu').numpy()

    if batch_size == 1:
        anomaly_map = anomaly_map[0]
        a_map_list = [a[0] for a in a_map_list]

    return anomaly_map, a_map_list


def cal_l2dis(fs_list, ft_list, out_size=256, amap_mode='mul'):
    """与 ``cal_anomaly_map`` 逻辑一致，但用逐通道 L2 距离代替余弦距离。"""
    # 处理 out_size：统一转为 tuple
    if isinstance(out_size, int):
        out_size_tuple = (out_size, out_size)
    else:
        out_size_tuple = out_size

    # 获取 batch size
    batch_size = fs_list[0].shape[0]

    if amap_mode == 'mul':
        anomaly_map = np.ones((batch_size, *out_size_tuple))
    else:
        anomaly_map = np.zeros((batch_size, *out_size_tuple))

    a_map_list = []
    for i in range(len(fs_list)):
        fs = fs_list[i].float()
        ft = ft_list[i].float()
        # 逐通道 L2 范数（||fs - ft||_2），dim=1 为通道维
        a_map = torch.norm(fs - ft, p=2, dim=1, keepdim=True)
        a_map = F.interpolate(a_map, size=out_size_tuple, mode='bilinear', align_corners=True)
        a_map = a_map[:, 0, :, :].float().to('cpu').detach().numpy()  # [B, H, W]
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map

    # 如果 batch_size=1，返回 [H, W]（兼容旧代码）
    if batch_size == 1:
        anomaly_map = anomaly_map[0]
        a_map_list = [a[0] for a in a_map_list]

    return anomaly_map, a_map_list


def min_max_norm(image):
    """把 2D 图像线性映射到 [0, 1] 区间，便于可视化。"""
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)


def cvt2heatmap(gray):
    """灰度图 -> JET 色彩空间的伪彩热图（供人眼观察）。"""
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap


def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> None:
    """计算 PRO（Per-Region Overlap）曲线在 FPR ∈ [0, 0.3] 区间的 AUC。

    PRO 指标的思想：对每个异常区域（连通组件）独立计算检出率，
    最后对所有区域取平均，避免大块异常区域把指标"带偏"。

    参数：
        masks : 所有测试样本的二值 gt 掩码，shape (N, H, W)，元素只能是 {0, 1}
        amaps : 对应的异常热图，shape (N, H, W)，数值任意
        num_th: 阈值采样数，默认 200
    返回：
        在 FPR ∈ [0, 0.3] 范围内归一化后的 PRO-AUC（标量）。
        数据不足或异常情况返回 0.0。
    """

    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=np.bool_)

    # 在热图的最小/最大值之间均匀取 num_th 个阈值
    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        # 大于阈值 = 预测异常
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        # 对每张图的每个连通异常区域，计算被正确检出的比例
        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)  # 区域检出率

        # 假正率：背景被预测为异常的比例
        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        df = df._append({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)

    # 只关注 FPR < 0.3 的部分，并把 FPR 归一化到 [0, 1] 便于求 AUC
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    # 防御：空表/全 0 等退化情况直接返回 0
    if df.empty is True or df.size == 0 or df["fpr"].max() == 0 or df.isnull().values.any() is True or len(df) <= 1:
        return 0.0

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc


def compute_pro_1(masks: ndarray, amaps: ndarray, num_th: int = 200) -> None:
    """更严格版 PRO-AUC：只看 FPR ∈ [0, 0.01] 区间（低假正率下的检出能力）。

    其余逻辑与 ``compute_pro`` 完全一致，仅阈值裁剪从 0.3 改为 0.01。
    这个指标常用于工业检测，因为实际生产中背景误报成本极高。
    """

    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=np.bool_)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        df = df._append({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)

    # 关键区别：只看 FPR < 0.01
    df = df[df["fpr"] < 0.01]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    if df.empty is True or df.size == 0 or df["fpr"].max() == 0 or df.isnull().values.any() is True or len(df) <= 1:
        return 0.0

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc
