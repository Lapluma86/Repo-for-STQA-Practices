# -*- coding: utf-8 -*-
"""
Stage B 占位：视角条件化的 FiLM 模块（每视角一组 gamma/beta）。

设计要点：
    - 参数量极小：num_views * num_channels * 2
    - 初始化为 (gamma=1, beta=0)，等价于"不做条件化"，保证从 Stage A
      的解码器权重热启动时训练稳定。
"""

import torch
import torch.nn as nn


class ViewFiLM(nn.Module):
    """对解码器某一层的特征做 per-view 仿射变换：feat = gamma_v * feat + beta_v"""

    def __init__(self, num_views: int, num_channels: int):
        super().__init__()
        self.num_views = num_views
        self.num_channels = num_channels
        self.gamma = nn.Parameter(torch.ones(num_views, num_channels))
        self.beta = nn.Parameter(torch.zeros(num_views, num_channels))

    def forward(self, feat: torch.Tensor, view_id: torch.Tensor) -> torch.Tensor:
        # feat: [B, C, H, W]; view_id: [B] long
        gamma = self.gamma[view_id].unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        beta = self.beta[view_id].unsqueeze(-1).unsqueeze(-1)
        return gamma * feat + beta
