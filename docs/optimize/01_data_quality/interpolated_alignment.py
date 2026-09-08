"""
混合时间对齐策略

根据特征类型选择不同的对齐方式：
- 离散特征（SSL嵌入、MFCC）：最近邻采样，保持特征完整性
- 连续几何特征（头部姿态、身体姿态）：线性插值，保持运动平滑性

优势：
1. 连续特征的时序平滑性提高
2. 减少抖动噪声
3. 更好地捕捉运动模式

使用示例：
    from interpolated_alignment import align_to_grid_interpolated

    aligned_feats, aligned_masks, grid_ms, T = align_to_grid_interpolated(
        all_groups,
        grid_step_ms=40.0,
        tolerance_ms=25.0,
        interpolate_features={"headpose_geom", "body_pose", "global_motion"}
    )
"""
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np


class SequenceData(NamedTuple):
    """时序特征数据容器（与feature_io.py保持一致）"""
    features: np.ndarray
    timestamps_ms: np.ndarray
    valid_mask: np.ndarray


def _nearest_indices(grid: np.ndarray, timestamps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    为统一时间网格的每个点，找到原始时间戳中最近的索引

    参数：
        grid: 统一时间网格，如[0, 40, 80, 120, ...]ms
        timestamps: 原始特征的时间戳，如[0, 33.3, 66.7, ...]ms（30fps视频）

    返回：
        best_idx: 最近邻索引数组
        best_dist: 最近邻距离数组（用于判断是否超出容差）
    """
    idx = np.searchsorted(timestamps, grid, side="left")
    idx = np.clip(idx, 0, len(timestamps) - 1)

    idx_left = np.clip(idx - 1, 0, len(timestamps) - 1)
    dist_right = np.abs(grid - timestamps[idx])
    dist_left = np.abs(grid - timestamps[idx_left])
    use_left = dist_left < dist_right
    best_idx = np.where(use_left, idx_left, idx)
    best_dist = np.where(use_left, dist_left, dist_right)
    return best_idx, best_dist


def _interpolate_features(
    grid: np.ndarray,
    timestamps: np.ndarray,
    features: np.ndarray,
) -> np.ndarray:
    """
    使用线性插值对齐特征到统一网格

    参数：
        grid: 统一时间网格，形状(T_grid,)
        timestamps: 原始时间戳，形状(T_orig,)
        features: 原始特征，形状(T_orig, D)

    返回：
        插值后的特征，形状(T_grid, D)

    实现细节：
        对每个特征维度独立进行线性插值，保持运动轨迹的连续性。
    """
    T_grid, D = len(grid), features.shape[1]
    aligned = np.zeros((T_grid, D), dtype=features.dtype)

    for d in range(D):
        # 对每个维度进行线性插值
        aligned[:, d] = np.interp(grid, timestamps, features[:, d])

    return aligned


def align_to_grid_interpolated(
    groups: dict[str, SequenceData],
    grid_step_ms: float = 40.0,
    tolerance_ms: float = 25.0,
    interpolate_features: set[str] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, int]:
    """
    混合时间对齐：根据特征类型选择对齐方式

    参数：
        groups: 特征组字典，键为"modality/feature_name"，值为SequenceData
        grid_step_ms: 统一时间网格步长（毫秒）
        tolerance_ms: 最近邻匹配容差（毫秒）
        interpolate_features: 需要插值的特征名称集合（不含模态前缀）
                             如{"headpose_geom", "body_pose", "global_motion"}
                             未指定的特征使用最近邻采样

    返回：
        aligned_feats: 对齐后的特征字典，键为"modality/feature_name"，值为(T, D)数组
        aligned_masks: 对齐后的掩码字典，键为"modality/feature_name"，值为(T,)布尔数组
        grid: 统一时间网格，形状(T,)
        T: 对齐后的序列长度

    特征类型建议：
        插值适用：
        - headpose_geom: 头部姿态（欧拉角、位置）
        - body_pose: 身体姿态（关键点坐标）
        - global_motion: 全局运动（光流统计）

        最近邻适用：
        - mel_mfcc: 梅尔频谱和MFCC（离散频谱特征）
        - ssl_embed: 自监督嵌入（高维语义特征）
        - face_behavior: 面部行为单元（离散动作单元）
        - vad: 语音活动检测（二值信号）

    示例：
        aligned_feats, aligned_masks, grid_ms, T = align_to_grid_interpolated(
            all_groups,
            grid_step_ms=40.0,
            tolerance_ms=25.0,
            interpolate_features={"headpose_geom", "body_pose", "global_motion"}
        )
    """
    if not groups:
        raise ValueError("No feature groups supplied for alignment")

    if interpolate_features is None:
        interpolate_features = set()

    # 计算全局时间范围
    t_min = min(seq.timestamps_ms[0] for seq in groups.values())
    t_max = max(seq.timestamps_ms[-1] for seq in groups.values())

    # 生成统一时间网格
    grid = np.arange(t_min, t_max + grid_step_ms * 0.5, grid_step_ms)
    T = len(grid)

    aligned_feats: dict[str, np.ndarray] = {}
    aligned_masks: dict[str, np.ndarray] = {}

    for name, seq in groups.items():
        # 提取特征名称（去除模态前缀）
        # 例如："audio/mel_mfcc" → "mel_mfcc"
        feature_name = name.split("/", 1)[1] if "/" in name else name

        # 根据特征类型选择对齐方式
        if feature_name in interpolate_features:
            # 策略1：线性插值（适用于连续几何特征）
            aligned_feats[name] = _interpolate_features(
                grid, seq.timestamps_ms, seq.features
            )

            # 插值后的掩码：基于最近邻距离判断
            best_idx, best_dist = _nearest_indices(grid, seq.timestamps_ms)
            within = best_dist <= tolerance_ms
            aligned_masks[name] = seq.valid_mask[best_idx] & within

        else:
            # 策略2：最近邻采样（适用于离散特征）
            best_idx, best_dist = _nearest_indices(grid, seq.timestamps_ms)
            within = best_dist <= tolerance_ms

            aligned_feats[name] = seq.features[best_idx]
            aligned_masks[name] = seq.valid_mask[best_idx] & within

    return aligned_feats, aligned_masks, grid, T


def compute_smoothness_metric(features: np.ndarray) -> float:
    """
    计算特征序列的平滑度指标

    参数：
        features: 特征序列，形状(T, D)

    返回：
        平滑度分数，范围[0, 1]，越高越平滑

    实现：
        使用一阶差分的标准差衡量抖动程度，归一化到[0, 1]。
        平滑的序列差分较小，抖动的序列差分较大。

    使用场景：
        评估对齐策略的效果，对比最近邻和插值的平滑度差异。

    示例：
        # 对比两种对齐方式
        nearest_smoothness = compute_smoothness_metric(nearest_aligned)
        interp_smoothness = compute_smoothness_metric(interp_aligned)
        print(f"插值平滑度提升: {(interp_smoothness - nearest_smoothness) * 100:.1f}%")
    """
    if len(features) < 2:
        return 1.0

    # 计算一阶差分（相邻帧的变化）
    diff = np.diff(features, axis=0)  # 形状: (T-1, D)

    # 计算差分的标准差（衡量抖动程度）
    diff_std = np.std(diff)

    # 归一化到[0, 1]：使用sigmoid函数
    # 差分越小（越平滑）→ 分数越高
    smoothness = 1.0 / (1.0 + diff_std)

    return float(smoothness)


def compare_alignment_strategies(
    seq: SequenceData,
    grid_step_ms: float = 40.0,
) -> dict[str, float]:
    """
    对比最近邻和插值两种对齐策略的效果

    参数：
        seq: 原始序列数据
        grid_step_ms: 网格步长

    返回：
        对比结果字典，包含：
        - nearest_smoothness: 最近邻对齐的平滑度
        - interp_smoothness: 插值对齐的平滑度
        - smoothness_gain: 平滑度提升（百分比）

    使用场景：
        在决定是否对某个特征使用插值前，先评估效果。

    示例：
        comparison = compare_alignment_strategies(headpose_seq)
        if comparison['smoothness_gain'] > 20:
            print("建议对该特征使用插值")
    """
    # 生成网格
    t_min, t_max = seq.timestamps_ms[0], seq.timestamps_ms[-1]
    grid = np.arange(t_min, t_max + grid_step_ms * 0.5, grid_step_ms)

    # 最近邻对齐
    best_idx, _ = _nearest_indices(grid, seq.timestamps_ms)
    nearest_aligned = seq.features[best_idx]

    # 插值对齐
    interp_aligned = _interpolate_features(grid, seq.timestamps_ms, seq.features)

    # 计算平滑度
    nearest_smoothness = compute_smoothness_metric(nearest_aligned)
    interp_smoothness = compute_smoothness_metric(interp_aligned)

    # 计算提升
    gain = (interp_smoothness - nearest_smoothness) / nearest_smoothness * 100

    return {
        "nearest_smoothness": nearest_smoothness,
        "interp_smoothness": interp_smoothness,
        "smoothness_gain": gain,
    }


if __name__ == "__main__":
    # 测试示例：模拟头部姿态数据
    print("=== 混合对齐策略测试 ===\n")

    # 模拟30fps视频的头部姿态数据（33.3ms间隔）
    T_orig = 300  # 10秒视频
    timestamps_orig = np.arange(T_orig) * 33.3  # 0, 33.3, 66.7, ...

    # 模拟平滑的头部旋转（正弦波 + 小噪声）
    t_sec = timestamps_orig / 1000.0
    yaw = 30 * np.sin(2 * np.pi * 0.5 * t_sec)  # 0.5Hz旋转
    pitch = 15 * np.cos(2 * np.pi * 0.3 * t_sec)  # 0.3Hz点头
    roll = 10 * np.sin(2 * np.pi * 0.2 * t_sec)  # 0.2Hz侧倾

    # 添加测量噪声
    noise_level = 2.0
    yaw += np.random.randn(T_orig) * noise_level
    pitch += np.random.randn(T_orig) * noise_level
    roll += np.random.randn(T_orig) * noise_level

    features_orig = np.stack([yaw, pitch, roll], axis=1)  # (T_orig, 3)
    valid_mask = np.ones(T_orig, dtype=bool)

    seq = SequenceData(
        features=features_orig,
        timestamps_ms=timestamps_orig,
        valid_mask=valid_mask,
    )

    # 对比两种对齐策略
    comparison = compare_alignment_strategies(seq, grid_step_ms=40.0)

    print("对齐策略对比（头部姿态）:")
    print(f"  最近邻平滑度: {comparison['nearest_smoothness']:.4f}")
    print(f"  插值平滑度: {comparison['interp_smoothness']:.4f}")
    print(f"  平滑度提升: {comparison['smoothness_gain']:.1f}%")

    if comparison['smoothness_gain'] > 10:
        print("\n✓ 建议：该特征适合使用插值对齐")
    else:
        print("\n✗ 建议：该特征使用最近邻对齐即可")
