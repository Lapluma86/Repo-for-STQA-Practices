"""
swa.py — Stochastic Weight Averaging（随机权重平均）

问题诊断：
    当前训练使用 CosineAnnealingLR 衰减到 1e-6，然后靠 EarlyStopping
    在验证指标最好的 epoch 保存检查点。

    这种做法的问题：
    1. 单个 checkpoint 的方差大：不同 seed 下最佳 epoch 的 QWK 可能差 5%+
    2. 模型可能停在 loss landscape 的尖锐最小值 → 泛化差
    3. CosineAnnealing 末尾 LR 极低，模型几乎不更新 → 浪费训练时间

改进方案 — SWA（Stochastic Weight Averaging）：
    在训练后期（如最后 25% 的 epoch），将多个 epoch 的权重做指数平均：
        w_swa = (1-α) × w_swa + α × w_current

    SWA 的理论基础：
    1. 平均多个权重 → 等效于 ensemble，降低方差
    2. 平均权重倾向于找到 loss landscape 的平坦区域 → 更好的泛化
    3. 几乎不增加训练开销（只需额外存储一份权重副本）
    4. 研究表明在分类任务上通常能提升 1~3%

    SWA 的关键参数：
    - swa_start: 从第几个 epoch 开始累积（推荐总 epoch 的 75%）
    - swa_lr: SWA 阶段的学习率（推荐比最终 LR 高 10x，如 1e-5）

预期提升：QWK +1~3%，减少不同 seed 间的方差

实施优先级：P1
"""
from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn


class SWAWrapper:
    """
    SWA 权重管理器

    使用方式：
        1. 正常训练到 swa_start epoch
        2. 从 swa_start 开始，每个 epoch 结束后调用 update()
        3. 训练结束后调用 apply() 将平均权重写回模型
        4. 在平均权重上做一轮 BatchNorm 统计量更新（如果有 BN）

    与 PyTorch 自带 torch.optim.swa_utils.AveragedModel 的区别：
        - PyTorch 版本需要改变训练循环结构（两阶段 scheduler）
        - 本实现是非侵入式的：只需要在训练循环中加两行代码

    参数:
        model:     需要做 SWA 的模型
        swa_start: 从第几个 epoch 开始累积
    """

    def __init__(self, model: nn.Module, swa_start: int = 20):
        self.model = model
        self.swa_start = swa_start
        self.n_averaged = 0

        # 深拷贝初始权重作为起点
        self.avg_state = None

    def update(self, epoch: int) -> bool:
        """
        在每个 epoch 结束后调用。如果 epoch >= swa_start，累积权重。

        返回:
            True 表示已累积，False 表示尚未开始
        """
        if epoch < self.swa_start:
            return False

        current_state = self.model.state_dict()

        if self.avg_state is None:
            # 第一次累积：直接复制当前权重
            self.avg_state = {k: v.clone().float() for k, v in current_state.items()}
            self.n_averaged = 1
        else:
            # 增量平均：new_avg = (n × old_avg + current) / (n+1)
            self.n_averaged += 1
            for k in self.avg_state:
                self.avg_state[k] += (current_state[k].float() - self.avg_state[k]) / self.n_averaged

        return True

    def apply(self) -> None:
        """
        将平均权重写回模型。

        注意：应用后需要用一个 epoch 的训练数据更新 BatchNorm 统计量
        （如果模型中有 BatchNorm 层的话）。
        当前骨干网络用的是 LayerNorm，不需要这一步。
        """
        if self.avg_state is None:
            return

        # 将 float64 平均权重转回模型的原始 dtype
        model_state = self.model.state_dict()
        for k in self.avg_state:
            model_state[k] = self.avg_state[k].to(model_state[k].dtype)

        self.model.load_state_dict(model_state)

    @property
    def is_active(self) -> bool:
        """是否已经开始累积"""
        return self.n_averaged > 0


def update_bn_stats(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    n_batches: int = 50,
) -> None:
    """
    在 SWA 平均权重上更新 BatchNorm 统计量

    为什么需要？
        SWA 平均了多个 epoch 的权重 → 中间特征的分布会与任何单个 epoch 不同
        → BatchNorm 的 running_mean/running_var 需要重新计算

    当前模型用 LayerNorm 没有这个问题，但如果将来加入 BN 层则需要此函数。

    参数:
        model:     SWA 后的模型
        loader:    训练数据加载器
        device:    设备
        n_batches: 用多少个 batch 更新统计量（默认 50 个 batch 足够）
    """
    was_training = model.training
    model.train()  # BN 在 train 模式下更新统计量

    # 重置所有 BN 层的统计量
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.reset_running_stats()

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            # 前向传播，让 BN 层更新统计量
            if isinstance(batch, dict) and "flat_batch" in batch:
                from common.runner import _to_device
                flat_batch = _to_device(batch["flat_batch"], device)
                session_valid = batch["session_valid"].to(device)
                B = batch["n_participants"]
                model(flat_batch, B, session_valid)

    if not was_training:
        model.eval()


# ============================================================
# 集成示例
# ============================================================
#
# # 在 runner.py 的 main() 中：
#
# swa_start = int(epochs * 0.75)  # 最后 25% 的 epoch 开始累积
# swa = SWAWrapper(grouped_model, swa_start=swa_start)
# swa_head = SWAWrapper(task_head, swa_start=swa_start)
#
# for epoch in range(1, epochs + 1):
#     # ... 正常训练和验证 ...
#
#     # 每个 epoch 结束后累积 SWA
#     swa.update(epoch)
#     swa_head.update(epoch)
#
# # 训练结束后，应用 SWA 权重
# if swa.is_active:
#     log.info(f"Applying SWA (averaged {swa.n_averaged} checkpoints)")
#     swa.apply()
#     swa_head.apply()
#
#     # 用 SWA 权重重新评估验证集
#     swa_metrics = validate_grouped(grouped_model, task_head, val_loader, ...)
#     log.info(f"SWA metrics: QWK={swa_metrics['mean_qwk']:.4f}")
#
#     # 如果 SWA 更好，保存 SWA 权重
#     if swa_metrics["primary_metric"] > best_metric:
#         save_checkpoint(run_dirs["checkpoints"] / "best_swa.pt", ...)
