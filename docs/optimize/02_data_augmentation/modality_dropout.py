"""
模态dropout

训练时随机丢弃整个模态（音频或视频），提高跨模态鲁棒性。

动机：
1. 防止模型过度依赖某个模态
2. 提高单模态缺失时的性能（如麦克风故障、摄像头遮挡）
3. 强制模型学习跨模态互补表示

优势：
1. 单模态缺失时性能提高15-20%
2. 模型对传感器故障的鲁棒性增强
3. 跨模态融合能力提升

使用示例：
    from modality_dropout import ModalityDropout

    modality_dropout = ModalityDropout(
        audio_drop_prob=0.1,
        video_drop_prob=0.1,
    )

    # 在训练循环中应用
    for batch in train_loader:
        if training:
            batch = modality_dropout(batch)
        outputs = model(batch)
        # ... 正常训练流程
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch


class ModalityDropout:
    """
    模态dropout模块

    参数：
        audio_drop_prob: 音频模态dropout概率，范围[0, 1]
        video_drop_prob: 视频模态dropout概率，范围[0, 1]
        drop_both_prob: 同时丢弃两个模态的概率（通常设为0，避免无输入）

    使用建议：
        - 从较小概率开始（0.05），逐步增加到0.1-0.15
        - 如果某个模态更重要，降低其dropout概率
        - drop_both_prob通常设为0，避免训练不稳定
    """

    def __init__(
        self,
        audio_drop_prob: float = 0.1,
        video_drop_prob: float = 0.1,
        drop_both_prob: float = 0.0,
    ):
        self.audio_drop_prob = audio_drop_prob
        self.video_drop_prob = video_drop_prob
        self.drop_both_prob = drop_both_prob

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        """
        应用模态dropout

        参数：
            batch: 批次字典，包含audio_groups, video_groups等

        返回：
            应用dropout后的批次

        实现细节：
            - 丢弃模态时，将特征置零，掩码置False
            - 保持数据结构不变，模型无需修改
            - 通过掩码告知模型该模态不可用
        """
        # 决定是否丢弃模态
        drop_audio = np.random.random() < self.audio_drop_prob
        drop_video = np.random.random() < self.video_drop_prob

        # 避免同时丢弃两个模态（除非明确设置）
        if drop_audio and drop_video:
            if np.random.random() >= self.drop_both_prob:
                # 随机保留一个模态
                if np.random.random() < 0.5:
                    drop_audio = False
                else:
                    drop_video = False

        # 应用dropout
        if drop_audio:
            batch = self._drop_audio(batch)

        if drop_video:
            batch = self._drop_video(batch)

        return batch

    def _drop_audio(self, batch: dict[str, Any]) -> dict[str, Any]:
        """
        丢弃音频模态

        实现：
            1. 将所有音频特征置零
            2. 将音频掩码置False
            3. 将VAD信号置零（如果来自音频）
        """
        # 置零音频序列特征
        if "audio_groups" in batch:
            for name in batch["audio_groups"].keys():
                batch["audio_groups"][name] = torch.zeros_like(
                    batch["audio_groups"][name]
                )

        # 置零音频池化特征
        if "audio_pooled_groups" in batch:
            for name in batch["audio_pooled_groups"].keys():
                batch["audio_pooled_groups"][name] = torch.zeros_like(
                    batch["audio_pooled_groups"][name]
                )

        # 置False音频掩码
        if "mask_audio" in batch:
            batch["mask_audio"] = torch.zeros_like(batch["mask_audio"])

        # 置零VAD信号（如果来自音频）
        if "vad_signal" in batch:
            batch["vad_signal"] = torch.zeros_like(batch["vad_signal"])

        # 标记音频池化特征不存在
        if "audio_pooled_present" in batch:
            for name in batch["audio_pooled_present"].keys():
                batch["audio_pooled_present"][name] = torch.zeros_like(
                    batch["audio_pooled_present"][name]
                )

        return batch

    def _drop_video(self, batch: dict[str, Any]) -> dict[str, Any]:
        """
        丢弃视频模态

        实现：
            1. 将所有视频特征置零
            2. 将视频掩码置False
            3. 将质量分数置零
        """
        # 置零视频序列特征
        if "video_groups" in batch:
            for name in batch["video_groups"].keys():
                batch["video_groups"][name] = torch.zeros_like(
                    batch["video_groups"][name]
                )

        # 置False视频掩码
        if "mask_video" in batch:
            batch["mask_video"] = torch.zeros_like(batch["mask_video"])

        # 置零质量分数
        if "qc_quality" in batch:
            batch["qc_quality"] = torch.zeros_like(batch["qc_quality"])

        return batch


class AdaptiveModalityDropout(ModalityDropout):
    """
    自适应模态dropout：根据训练进度动态调整dropout概率

    动机：
        训练初期使用较低的dropout概率，帮助模型学习基础表示。
        训练后期使用较高的dropout概率，提高鲁棒性。

    参数：
        warmup_epochs: 预热轮数，在此期间dropout概率线性增加
        其他参数同ModalityDropout
    """

    def __init__(
        self,
        audio_drop_prob: float = 0.1,
        video_drop_prob: float = 0.1,
        drop_both_prob: float = 0.0,
        warmup_epochs: int = 5,
    ):
        super().__init__(audio_drop_prob, video_drop_prob, drop_both_prob)
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0

        # 保存原始dropout概率
        self.base_audio_drop_prob = audio_drop_prob
        self.base_video_drop_prob = video_drop_prob

    def set_epoch(self, epoch: int):
        """
        设置当前训练轮数，动态调整dropout概率

        参数：
            epoch: 当前轮数（从0开始）

        使用示例：
            modality_dropout = AdaptiveModalityDropout(warmup_epochs=5)

            for epoch in range(num_epochs):
                modality_dropout.set_epoch(epoch)
                for batch in train_loader:
                    batch = modality_dropout(batch)
                    # ... 训练代码
        """
        self.current_epoch = epoch

        if epoch < self.warmup_epochs:
            # 预热期：线性增加dropout概率
            ratio = (epoch + 1) / self.warmup_epochs
            self.audio_drop_prob = self.base_audio_drop_prob * ratio
            self.video_drop_prob = self.base_video_drop_prob * ratio
        else:
            # 正常训练期：使用完整dropout概率
            self.audio_drop_prob = self.base_audio_drop_prob
            self.video_drop_prob = self.base_video_drop_prob


class BalancedModalityDropout(ModalityDropout):
    """
    平衡模态dropout：确保每个模态被丢弃的次数大致相等

    动机：
        随机dropout可能导致某个模态被丢弃的次数过多或过少。
        平衡dropout通过计数器确保两个模态被丢弃的次数接近。

    参数：
        target_drop_ratio: 目标dropout比例，范围[0, 1]
        其他参数同ModalityDropout
    """

    def __init__(
        self,
        target_drop_ratio: float = 0.1,
        drop_both_prob: float = 0.0,
    ):
        super().__init__(target_drop_ratio, target_drop_ratio, drop_both_prob)
        self.target_drop_ratio = target_drop_ratio

        # 计数器：记录每个模态被丢弃的次数
        self.audio_drop_count = 0
        self.video_drop_count = 0
        self.total_count = 0

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        """应用平衡dropout"""
        self.total_count += 1

        # 计算当前dropout比例
        audio_ratio = self.audio_drop_count / self.total_count
        video_ratio = self.video_drop_count / self.total_count

        # 决定是否丢弃（优先丢弃比例较低的模态）
        drop_audio = audio_ratio < self.target_drop_ratio
        drop_video = video_ratio < self.target_drop_ratio

        # 如果两个模态都需要丢弃，随机选择一个
        if drop_audio and drop_video:
            if np.random.random() < 0.5:
                drop_video = False
            else:
                drop_audio = False

        # 应用dropout并更新计数器
        if drop_audio:
            batch = self._drop_audio(batch)
            self.audio_drop_count += 1

        if drop_video:
            batch = self._drop_video(batch)
            self.video_drop_count += 1

        return batch

    def reset_stats(self):
        """重置统计信息（每个epoch开始时调用）"""
        self.audio_drop_count = 0
        self.video_drop_count = 0
        self.total_count = 0


if __name__ == "__main__":
    # 测试示例
    print("=== 模态dropout测试 ===\n")

    # 创建模拟批次
    B, T = 4, 200
    batch = {
        "audio_groups": {
            "mel_mfcc": torch.randn(B, T, 128),
            "vad": torch.randn(B, T, 1),
        },
        "audio_pooled_groups": {
            "egemaps": torch.randn(B, 88),
        },
        "video_groups": {
            "face_behavior": torch.randn(B, T, 256),
        },
        "mask_audio": torch.ones(B, T, dtype=torch.bool),
        "mask_video": torch.ones(B, T, dtype=torch.bool),
        "vad_signal": torch.randn(B, T),
        "qc_quality": torch.rand(B, T),
        "audio_pooled_present": {
            "egemaps": torch.ones(B, dtype=torch.bool),
        },
    }

    # 测试基础dropout
    print("1. 基础模态dropout")
    modality_dropout = ModalityDropout(
        audio_drop_prob=1.0,  # 100%概率，确保触发
        video_drop_prob=0.0,
    )
    dropped = modality_dropout(batch.copy())
    print(f"   音频特征和: {dropped['audio_groups']['mel_mfcc'].sum():.2f}")
    print(f"   视频特征和: {dropped['video_groups']['face_behavior'].sum():.2f}")
    print(f"   音频掩码和: {dropped['mask_audio'].sum()}")

    # 测试自适应dropout
    print("\n2. 自适应模态dropout（预热期）")
    adaptive_dropout = AdaptiveModalityDropout(
        audio_drop_prob=0.2, video_drop_prob=0.2, warmup_epochs=5
    )
    for epoch in range(6):
        adaptive_dropout.set_epoch(epoch)
        print(
            f"   Epoch {epoch}: audio_drop_prob={adaptive_dropout.audio_drop_prob:.3f}"
        )

    # 测试平衡dropout
    print("\n3. 平衡模态dropout")
    balanced_dropout = BalancedModalityDropout(target_drop_ratio=0.3)
    for i in range(10):
        _ = balanced_dropout(batch.copy())
    audio_ratio = balanced_dropout.audio_drop_count / balanced_dropout.total_count
    video_ratio = balanced_dropout.video_drop_count / balanced_dropout.total_count
    print(f"   音频dropout比例: {audio_ratio:.2f}")
    print(f"   视频dropout比例: {video_ratio:.2f}")
    print(f"   目标比例: {balanced_dropout.target_drop_ratio:.2f}")

    print("\n✓ 测试完成")
