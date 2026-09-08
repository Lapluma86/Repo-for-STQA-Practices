# -*- coding: utf-8 -*-
"""Stage B 占位：可学习的视角 prompt embedding（拼接到 bottleneck token 序列）。"""

import torch
import torch.nn as nn


class ViewPrompt(nn.Module):
    """每视角一组可学习 token，长度 = prompt_len，维度 = dim。"""

    def __init__(self, num_views: int, prompt_len: int, dim: int):
        super().__init__()
        self.prompts = nn.Parameter(torch.zeros(num_views, prompt_len, dim))
        nn.init.trunc_normal_(self.prompts, std=0.02)

    def forward(self, view_id: torch.Tensor) -> torch.Tensor:
        # 返回 [B, prompt_len, dim]
        return self.prompts[view_id]
