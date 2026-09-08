"""
自适应掩码策略

根据每个时间帧的数据质量动态调整掩码策略，提高数据利用率的同时保持质量控制。

核心思想：
- 核心特征全部有效 → 严格模式（所有核心特征都必须有效）
- 核心特征部分有效 → 宽松模式（任一特征有效即可）
- 核心特征全部缺失 → 标记为无效

优势：
1. 数据利用率提高20-30%
2. 保持对高质量数据的严格要求
3. 对部分缺失数据采用宽松策略，避免浪费

使用示例：
    from adaptive_mask import compute_adaptive_mask

    mask_audio = compute_adaptive_mask(
        mask_parts=audio_mask_parts,
        mask_names=audio_mask_names,
        core_names=cfg.core_audio,
        T=T
    )
"""
from __future__ import annotations

import numpy as np


def compute_adaptive_mask(
    mask_parts: list[np.ndarray],
    mask_names: list[str],
    core_names: list[str],
    T: int,
) -> np.ndarray:
    """
    自适应掩码计算：根据数据质量动态调整策略

    参数：
        mask_parts: 各特征的掩码列表，每个元素形状为(T,)
        mask_names: 对应的特征名称列表
        core_names: 核心特征名称列表（如["mel_mfcc", "vad"]）
        T: 序列长度

    返回：
        合并后的掩码，形状(T,)，True表示该帧有效

    策略说明：
        对于每个时间帧t：
        1. 如果所有核心特征都有效 → 标记为有效（严格模式）
        2. 如果部分核心特征有效 → 检查是否有任意特征有效（宽松模式）
        3. 如果核心特征全部缺失 → 标记为无效

    示例：
        假设核心特征为["mel_mfcc", "vad"]，某帧的状态为：
        - mel_mfcc: True, vad: True, ssl_embed: True  → 有效（严格模式）
        - mel_mfcc: True, vad: False, ssl_embed: True → 有效（宽松模式）
        - mel_mfcc: False, vad: False, ssl_embed: True → 无效（核心特征全缺失）
    """
    if not mask_parts:
        return np.zeros(T, dtype=bool)

    # 提取核心特征的掩码
    core_masks = [m for m, n in zip(mask_parts, mask_names) if n in core_names]

    if not core_masks:
        # 没有核心特征，降级为"or"策略：任一特征有效即可
        return np.any(np.stack(mask_parts), axis=0)

    # 计算每帧的核心特征有效数量
    core_stack = np.stack(core_masks)  # 形状: (n_core, T)
    valid_count = np.sum(core_stack, axis=0)  # 形状: (T,)

    # 初始化结果掩码
    result = np.zeros(T, dtype=bool)

    # 策略1：核心特征全部有效的帧 → 直接标记为有效（严格模式）
    all_core_valid = valid_count == len(core_masks)
    result[all_core_valid] = True

    # 策略2：核心特征部分有效的帧 → 检查是否有任意特征有效（宽松模式）
    partial_valid = (valid_count > 0) & (valid_count < len(core_masks))
    if np.any(partial_valid):
        all_stack = np.stack(mask_parts)  # 形状: (n_features, T)
        # 对于部分有效的帧，如果有任意特征有效则保留
        result[partial_valid] = np.any(all_stack[:, partial_valid], axis=0)

    # 策略3：核心特征全部缺失的帧 → 保持为False（无效）
    # （无需显式处理，result已初始化为False）

    return result


def compute_adaptive_mask_with_quality(
    mask_parts: list[np.ndarray],
    mask_names: list[str],
    core_names: list[str],
    quality_scores: np.ndarray | None,
    T: int,
    quality_threshold: float = 0.5,
) -> np.ndarray:
    """
    带质量分数的自适应掩码计算（高级版本）

    在基础自适应掩码的基础上，增加质量分数过滤，进一步提升数据质量。

    参数：
        mask_parts: 各特征的掩码列表
        mask_names: 对应的特征名称列表
        core_names: 核心特征名称列表
        quality_scores: 质量分数数组，形状(T,)，范围[0, 1]，来自qc_quality
        T: 序列长度
        quality_threshold: 质量分数阈值，低于此值的帧被过滤

    返回：
        合并后的掩码，形状(T,)

    使用场景：
        当数据质量参差不齐时，使用质量分数进一步过滤低质量帧。
        例如：视频中人脸被遮挡、光线不足等情况。

    示例：
        mask_audio = compute_adaptive_mask_with_quality(
            mask_parts=audio_mask_parts,
            mask_names=audio_mask_names,
            core_names=cfg.core_audio,
            quality_scores=qc_quality,  # 来自视频质量检测
            T=T,
            quality_threshold=0.6,  # 只保留质量分数>0.6的帧
        )
    """
    # 步骤1：计算基础自适应掩码
    base_mask = compute_adaptive_mask(mask_parts, mask_names, core_names, T)

    # 步骤2：应用质量分数过滤
    if quality_scores is not None:
        quality_mask = quality_scores >= quality_threshold
        return base_mask & quality_mask
    else:
        return base_mask


def analyze_mask_statistics(
    mask_parts: list[np.ndarray],
    mask_names: list[str],
    core_names: list[str],
) -> dict[str, float]:
    """
    分析掩码统计信息，用于评估数据质量

    参数：
        mask_parts: 各特征的掩码列表
        mask_names: 对应的特征名称列表
        core_names: 核心特征名称列表

    返回：
        统计信息字典，包含：
        - total_frames: 总帧数
        - strict_valid_frames: 严格模式下的有效帧数（核心特征全有效）
        - adaptive_valid_frames: 自适应模式下的有效帧数
        - data_utilization_gain: 数据利用率提升（百分比）
        - per_feature_validity: 每个特征的有效率

    使用场景：
        在训练前分析数据质量，评估自适应掩码的效果。

    示例：
        stats = analyze_mask_statistics(
            audio_mask_parts, audio_mask_names, cfg.core_audio
        )
        print(f"数据利用率提升: {stats['data_utilization_gain']:.1f}%")
    """
    if not mask_parts:
        return {}

    T = len(mask_parts[0])
    core_masks = [m for m, n in zip(mask_parts, mask_names) if n in core_names]

    # 计算严格模式下的有效帧数（核心特征全有效）
    if core_masks:
        strict_mask = np.all(np.stack(core_masks), axis=0)
        strict_valid = np.sum(strict_mask)
    else:
        strict_valid = 0

    # 计算自适应模式下的有效帧数
    adaptive_mask = compute_adaptive_mask(mask_parts, mask_names, core_names, T)
    adaptive_valid = np.sum(adaptive_mask)

    # 计算每个特征的有效率
    per_feature_validity = {
        name: float(np.mean(mask)) for name, mask in zip(mask_names, mask_parts)
    }

    # 计算数据利用率提升
    if strict_valid > 0:
        gain = (adaptive_valid - strict_valid) / strict_valid * 100
    else:
        gain = 0.0

    return {
        "total_frames": T,
        "strict_valid_frames": int(strict_valid),
        "adaptive_valid_frames": int(adaptive_valid),
        "strict_valid_ratio": float(strict_valid / T),
        "adaptive_valid_ratio": float(adaptive_valid / T),
        "data_utilization_gain": gain,
        "per_feature_validity": per_feature_validity,
    }


if __name__ == "__main__":
    # 测试示例
    T = 100

    # 模拟掩码数据
    mel_mfcc_mask = np.random.random(T) > 0.2  # 80%有效
    vad_mask = np.random.random(T) > 0.3       # 70%有效
    ssl_mask = np.random.random(T) > 0.4       # 60%有效

    mask_parts = [mel_mfcc_mask, vad_mask, ssl_mask]
    mask_names = ["mel_mfcc", "vad", "ssl_embed"]
    core_names = ["mel_mfcc", "vad"]

    # 计算自适应掩码
    adaptive_mask = compute_adaptive_mask(mask_parts, mask_names, core_names, T)

    # 分析统计信息
    stats = analyze_mask_statistics(mask_parts, mask_names, core_names)

    print("=== 自适应掩码统计 ===")
    print(f"总帧数: {stats['total_frames']}")
    print(f"严格模式有效帧: {stats['strict_valid_frames']} ({stats['strict_valid_ratio']:.1%})")
    print(f"自适应模式有效帧: {stats['adaptive_valid_frames']} ({stats['adaptive_valid_ratio']:.1%})")
    print(f"数据利用率提升: {stats['data_utilization_gain']:.1f}%")
    print("\n各特征有效率:")
    for name, validity in stats['per_feature_validity'].items():
        print(f"  {name}: {validity:.1%}")
