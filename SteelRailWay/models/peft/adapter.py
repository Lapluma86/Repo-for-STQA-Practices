# -*- coding: utf-8 -*-
"""Stage B 占位：bottleneck MLP Adapter（per-view 实例化）。待后续实现。"""

import torch
import torch.nn as nn


class BottleneckAdapter(nn.Module):
    """
    经典 Houlsby-style adapter：Down-proj -> 非线性 -> Up-proj，残差连接。
    通常以"每视角一份"的方式插入解码器各级。
    """

    def __init__(self, dim: int, bottleneck: int = 64):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] -> 转为 [B, H*W, C] 走 Linear
        b, c, h, w = x.shape
        z = x.flatten(2).transpose(1, 2)
        z = self.up(self.act(self.down(z)))
        z = z.transpose(1, 2).reshape(b, c, h, w)
        return x + z
