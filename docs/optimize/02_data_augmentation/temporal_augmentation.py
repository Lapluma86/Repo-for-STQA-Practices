"""
时序数据增强

实现两种时序增强策略，提升模型对时序变化的鲁棒性：
1. 时间掩码（Time Masking）：随机遮挡连续时间段，模拟注意力分散
2. 速度扰动（Speed Perturbation）：轻微加速/减速，增强时序鲁棒性

优势：
1. 防止过拟合特定时间模式
2. 提高对说话速度、动作速度变化的鲁棒性
3. 模拟真实场景中的数据缺失和时序变化

使用示例：
    from temporal_augmentation import TemporalAugmentation

    augmentation = TemporalAugmentation(
        time_mask_prob=0.15,
        speed_perturb_prob=0.3,
    )

    # 在数据集中集成
    train_dataset = MultimodalDataset(
        manifest_path="train.csv",
        cfg=cfg,
        split="train",
        augmentation=augmentation,
    )
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch


class TemporalAugmentation:
    """
    时序数据增强模块

    参数：
        time_mask_prob: 时间掩码概率，范围[0, 1]
        time_mask_max_ratio: 最大掩码比例，范围[0, 1]，相对于序列长度
        speed_perturb_prob: 速度扰动概率，范围[0, 1]
        speed_range: 速度范围，如(0.9, 1.1)表示0.9x-1.1x速度

    使用建议：
        - time_mask_prob: 0.1-0.2，过高会丢失太多信息
        - time_mask_max_ratio: 0.05-0.15，建议不超过15%
        - speed_perturb_prob: 0.2-0.4
        - speed_range: (0.9, 1.1)，过大会破坏时序模式
    """

    def __init__(
        self,
        time_mask_prob: float = 0.15,
        time_mask_max_ratio: float = 0.1,
        speed_perturb_prob: float = 0.3,
        speed_range: tuple[float, float] = (0.9, 1.1),
    ):
        self.time_mask_prob = time_mask_prob
        self.time_mask_max_ratio = time_mask_max_ratio
        self.speed_perturb_prob = speed_perturb_prob
        self.speed_range = speed_range

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        """
        应用时序增强

        参数：
            sample: 单个样本字典，包含audio_groups, video_groups等

        返回：
            增强后的样本字典
        """
        # 策略1：时间掩码
        if np.random.random() < self.time_mask_prob:
            sample = self._apply_time_mask(sample)

        # 策略2：速度扰动
        if np.random.random() < self.speed_perturb_prob:
            sample = self._apply_speed_perturb(sample)

        return sample

    def _apply_time_mask(self, sample: dict[str, Any]) -> dict[str, Any]:
        """
        时间掩码：随机遮挡连续时间段

        动机：
            模拟真实场景中的注意力分散、短暂遮挡等情况。
            例如：被试短暂低头看手机、摄像头被遮挡等。

        实现：
            随机选择一个连续时间段，将该段的掩码设为False。
            特征值保持不变，但通过掩码告知模型该段数据不可信。

        参数：
            sample: 样本字典

        返回：
            应用掩码后的样本
        """
        T = sample["seq_len"]
        max_mask_len = max(1, int(T * self.time_mask_max_ratio))
        mask_len = np.random.randint(1, max_mask_len + 1)
        mask_start = np.random.randint(0, max(1, T - mask_len + 1))

        # 将掩码区域设为False（标记为无效）
        sample["mask_audio"][mask_start : mask_start + mask_len] = False
        sample["mask_video"][mask_start : mask_start + mask_len] = False

        return sample

    def _apply_speed_perturb(self, sample: dict[str, Any]) -> dict[str, Any]:
        """
        速度扰动：轻微加速/减速整个序列

        动机：
            不同人的说话速度、动作速度不同。
            通过速度扰动，模型学习对速度变化的不变性。

        实现：
            通过重采样改变序列长度：
            - speed > 1.0：加速（序列变短）
            - speed < 1.0：减速（序列变长）

        注意：
            速度范围不宜过大，建议[0.9, 1.1]，否则会破坏时序模式。

        参数：
            sample: 样本字典

        返回：
            速度扰动后的样本
        """
        speed = np.random.uniform(*self.speed_range)
        T_old = sample["seq_len"]
        T_new = max(1, int(T_old / speed))

        # 生成重采样索引（线性插值）
        indices = np.linspace(0, T_old - 1, T_new)

        # 对所有序列特征进行重采样
        for key in ["audio_groups", "video_groups"]:
            if key not in sample:
                continue
            for name, feat in sample[key].items():
                # feat形状: (T_old, D)
                D = feat.shape[1]
                resampled = np.zeros((T_new, D), dtype=np.float32)

                for d in range(D):
                    # 对每个维度进行线性插值
                    resampled[:, d] = np.interp(
                        indices, np.arange(T_old), feat[:, d].numpy()
                    )

                sample[key][name] = torch.from_numpy(resampled)

        # 重采样掩码和辅助信号（使用最近邻）
        indices_int = np.round(indices).astype(int)
        indices_int = np.clip(indices_int, 0, T_old - 1)

        sample["mask_audio"] = sample["mask_audio"][indices_int]
        sample["mask_video"] = sample["mask_video"][indices_int]
        sample["vad_signal"] = sample["vad_signal"][indices_int]
        sample["qc_quality"] = sample["qc_quality"][indices_int]

        # 更新序列长度
        sample["seq_len"] = T_new

        return sample


class AdaptiveTemporalAugmentation(TemporalAugmentation):
    """
    自适应时序增强：根据训练进度动态调整增强强度

    动机：
        训练初期使用较弱的增强，帮助模型快速收敛。
        训练后期使用较强的增强，提高泛化能力。

    参数：
        warmup_epochs: 预热轮数，在此期间增强强度线性增加
        其他参数同TemporalAugmentation
    """

    def __init__(
        self,
        time_mask_prob: float = 0.15,
        time_mask_max_ratio: float = 0.1,
        speed_perturb_prob: float = 0.3,
        speed_range: tuple[float, float] = (0.9, 1.1),
        warmup_epochs: int = 5,
    ):
        super().__init__(
            time_mask_prob, time_mask_max_ratio, speed_perturb_prob, speed_range
        )
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0

        # 保存原始增强强度
        self.base_time_mask_prob = time_mask_prob
        self.base_speed_perturb_prob = speed_perturb_prob

    def set_epoch(self, epoch: int):
        """
        设置当前训练轮数，动态调整增强强度

        参数：
            epoch: 当前轮数（从0开始）

        使用示例：
            augmentation = AdaptiveTemporalAugmentation(warmup_epochs=5)

            for epoch in range(num_epochs):
                augmentation.set_epoch(epoch)
                for batch in train_loader:
                    # ... 训练代码
        """
        self.current_epoch = epoch

        if epoch < self.warmup_epochs:
            # 预热期：线性增加增强强度
            ratio = (epoch + 1) / self.warmup_epochs
            self.time_mask_prob = self.base_time_mask_prob * ratio
            self.speed_perturb_prob = self.base_speed_perturb_prob * ratio
        else:
            # 正常训练期：使用完整增强强度
            self.time_mask_prob = self.base_time_mask_prob
            self.speed_perturb_prob = self.base_speed_perturb_prob


class ConditionalTemporalAugmentation(TemporalAugmentation):
    """
    条件时序增强：根据样本特征选择性应用增强

    动机：
        不同样本的特征不同，应采用不同的增强策略：
        - 短序列：不使用时间掩码（避免丢失过多信息）
        - 低质量样本：不使用速度扰动（避免进一步降低质量）

    参数：
        min_len_for_mask: 应用时间掩码的最小序列长度
        min_quality_for_speed: 应用速度扰动的最小质量分数
        其他参数同TemporalAugmentation
    """

    def __init__(
        self,
        time_mask_prob: float = 0.15,
        time_mask_max_ratio: float = 0.1,
        speed_perturb_prob: float = 0.3,
        speed_range: tuple[float, float] = (0.9, 1.1),
        min_len_for_mask: int = 100,
        min_quality_for_speed: float = 0.5,
    ):
        super().__init__(
            time_mask_prob, time_mask_max_ratio, speed_perturb_prob, speed_range
        )
        self.min_len_for_mask = min_len_for_mask
        self.min_quality_for_speed = min_quality_for_speed

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        """应用条件增强"""
        # 条件1：序列足够长才使用时间掩码
        if (
            sample["seq_len"] >= self.min_len_for_mask
            and np.random.random() < self.time_mask_prob
        ):
            sample = self._apply_time_mask(sample)

        # 条件2：质量足够高才使用速度扰动
        mean_quality = float(sample["qc_quality"].mean())
        if (
            mean_quality >= self.min_quality_for_speed
            and np.random.random() < self.speed_perturb_prob
        ):
            sample = self._apply_speed_perturb(sample)

        return sample


if __name__ == "__main__":
    # 测试示例
    print("=== 时序数据增强测试 ===\n")

    # 创建模拟样本
    T = 200
    D_audio = 128
    D_video = 256

    sample = {
        "audio_groups": {
            "mel_mfcc": torch.randn(T, D_audio),
            "vad": torch.randn(T, 1),
        },
        "video_groups": {
            "face_behavior": torch.randn(T, D_video),
        },
        "mask_audio": torch.ones(T, dtype=torch.bool),
        "mask_video": torch.ones(T, dtype=torch.bool),
        "vad_signal": torch.randn(T),
        "qc_quality": torch.rand(T),
        "seq_len": T,
    }

    # 测试基础增强
    print("1. 基础时序增强")
    augmentation = TemporalAugmentation(
        time_mask_prob=1.0,  # 100%概率，确保触发
        speed_perturb_prob=0.0,
    )
    augmented = augmentation(sample.copy())
    print(f"   原始序列长度: {sample['seq_len']}")
    print(f"   增强后序列长度: {augmented['seq_len']}")
    print(f"   音频有效帧: {augmented['mask_audio'].sum()}/{T}")
    print(f"   视频有效帧: {augmented['mask_video'].sum()}/{T}")

    # 测试速度扰动
    print("\n2. 速度扰动增强")
    augmentation = TemporalAugmentation(
        time_mask_prob=0.0,
        speed_perturb_prob=1.0,
        speed_range=(0.8, 1.2),
    )
    augmented = augmentation(sample.copy())
    print(f"   原始序列长度: {sample['seq_len']}")
    print(f"   增强后序列长度: {augmented['seq_len']}")
    print(f"   长度变化: {(augmented['seq_len'] - T) / T * 100:.1f}%")

    # 测试自适应增强
    print("\n3. 自适应增强（预热期）")
    adaptive_aug = AdaptiveTemporalAugmentation(
        time_mask_prob=0.2, warmup_epochs=5
    )
    for epoch in range(6):
        adaptive_aug.set_epoch(epoch)
        print(
            f"   Epoch {epoch}: time_mask_prob={adaptive_aug.time_mask_prob:.3f}"
        )

    print("\n✓ 测试完成")
