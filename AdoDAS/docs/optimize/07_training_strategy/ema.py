"""
ema.py — 指数移动平均（Exponential Moving Average）

问题诊断：
    当前的训练保存策略是"验证集最佳 epoch 的 checkpoint"。
    问题：
    1. 单个 epoch 的权重波动大（尤其是小 batch 和高学习率时）
    2. 如果最佳 epoch 恰好在 loss landscape 的"尖锐谷底"，泛化性差
    3. 不同 seed 之间的最佳 epoch 方差大

改进方案 — EMA：
    维护一个模型参数的指数移动平均副本：
        θ_ema = β × θ_ema + (1-β) × θ_current

    其中 β = 0.999（典型值）。

    EMA 的优势：
    1. 天然平滑：等效于最近 ~1000 步的加权平均
    2. 效果类似 ensemble 但不增加推理成本
    3. 与 SWA 互补：SWA 是 epoch 级别均匀平均，EMA 是 step 级别指数平均

    为什么 EMA 比 SWA 在某些场景更好？
    - EMA 在每个训练步更新，更细粒度
    - EMA 对最近的权重给更高权重（指数衰减），适应性更强
    - 但 EMA 需要始终维护额外一份权重 → 双倍显存

    推荐组合：EMA (step-level) + SWA (epoch-level) + Top-K checkpoint

预期提升：QWK +1~2%，减少训练不稳定性

实施优先级：P1
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn


class EMA:
    """
    模型参数的指数移动平均

    用法：
        ema = EMA(model, decay=0.999)
        for step, batch in enumerate(loader):
            loss = model(batch)
            loss.backward()
            optimizer.step()
            ema.update()    # 每步更新 EMA

        # 验证时使用 EMA 权重
        with ema.apply_temporary():
            val_metrics = validate(model, val_loader)

        # 或者永久应用
        ema.apply_permanent()

    参数:
        model: 被跟踪的模型
        decay: 衰减系数 β（推荐 0.999~0.9999）
               β=0.999 → ~1000 步的平均窗口
               β=0.9999 → ~10000 步的平均窗口
        warmup_steps: 前 N 步不使用 EMA（让模型先稳定）
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        warmup_steps: int = 100,
    ):
        self.model = model
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.step_count = 0

        # 深拷贝当前权重作为 EMA 初始值
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

        # 保存原始权重（用于临时切换）
        self._backup = {}

    def update(self) -> None:
        """每个训练步调用一次"""
        self.step_count += 1

        # warmup 期间使用较小的 decay（更快追踪早期权重变化）
        if self.step_count <= self.warmup_steps:
            # 动态 decay：从 0 线性增长到目标 decay
            decay = min(self.decay, (1 + self.step_count) / (10 + self.step_count))
        else:
            decay = self.decay

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    # θ_ema = β × θ_ema + (1-β) × θ_current
                    self.shadow[name].mul_(decay).add_(param.data, alpha=1 - decay)

    def apply_permanent(self) -> None:
        """将 EMA 权重永久写入模型"""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.shadow:
                    param.data.copy_(self.shadow[name])

    class _TemporaryContext:
        """上下文管理器，临时使用 EMA 权重"""
        def __init__(self, ema: 'EMA'):
            self.ema = ema

        def __enter__(self):
            # 备份当前权重
            self.ema._backup = {}
            for name, param in self.ema.model.named_parameters():
                if name in self.ema.shadow:
                    self.ema._backup[name] = param.data.clone()
                    param.data.copy_(self.ema.shadow[name])
            return self.ema.model

        def __exit__(self, *args):
            # 恢复原始权重
            for name, param in self.ema.model.named_parameters():
                if name in self.ema._backup:
                    param.data.copy_(self.ema._backup[name])
            self.ema._backup = {}

    def apply_temporary(self) -> _TemporaryContext:
        """
        上下文管理器：临时使用 EMA 权重，退出后恢复

        用法：
            with ema.apply_temporary():
                val_metrics = validate(model, val_loader)
            # 退出后 model 恢复到训练权重
        """
        return self._TemporaryContext(self)


# ============================================================
# 集成示例
# ============================================================
#
# # 在 runner.py 的 main() 中：
#
# ema_model = EMA(grouped_model, decay=0.999)
# ema_head = EMA(task_head, decay=0.999)
#
# for epoch in range(1, epochs + 1):
#     for step, batch in enumerate(train_loader):
#         # ... 正常训练步 ...
#         optimizer.step()
#
#         # 每步更新 EMA
#         ema_model.update()
#         ema_head.update()
#
#     # 验证时临时使用 EMA 权重
#     with ema_model.apply_temporary(), ema_head.apply_temporary():
#         val_metrics_ema = validate_grouped(
#             grouped_model, task_head, val_loader, ...
#         )
#     log.info(f"EMA val QWK: {val_metrics_ema['mean_qwk']:.4f}")
#
#     # 同时验证原始权重
#     val_metrics = validate_grouped(grouped_model, task_head, val_loader, ...)
#
#     # 选更好的
#     if val_metrics_ema["primary_metric"] > val_metrics["primary_metric"]:
#         # 保存 EMA 版本
#         ema_model.apply_permanent()
#         ema_head.apply_permanent()
#         save_checkpoint(...)
#         # 注意：保存后需要恢复训练权重继续训练
#         # 这里简化处理，实际中可以用 apply_temporary
