# -*- coding: utf-8 -*-
"""Input-level affine PEFT for the rail depth branch."""

from __future__ import annotations

import torch
import torch.nn as nn


class DepthAffinePEFT(nn.Module):
    """Two-parameter depth adapter: ``D' = gain * D + bias``."""

    def __init__(self, init_gain: float = 1.0, init_bias: float = 0.0):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(float(init_gain), dtype=torch.float32))
        self.bias = nn.Parameter(torch.tensor(float(init_bias), dtype=torch.float32))

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        return self.gain.to(dtype=depth.dtype) * depth + self.bias.to(dtype=depth.dtype)

    @torch.no_grad()
    def reset_to_identity(self) -> None:
        self.gain.fill_(1.0)
        self.bias.fill_(0.0)

    def extra_repr(self) -> str:
        return f"gain={self.gain.item():.4f}, bias={self.bias.item():.4f}"


class DepthEncoderWithPEFT(nn.Module):
    """Wrap a frozen depth encoder so PEFT is applied before feature extraction."""

    def __init__(self, depth_encoder: nn.Module, peft: nn.Module | None = None):
        super().__init__()
        self.encoder = depth_encoder
        for param in self.encoder.parameters():
            param.requires_grad_(False)
        self.peft = peft if peft is not None else DepthAffinePEFT()

    def forward(self, depth: torch.Tensor):
        return self.encoder(self.peft(depth))

    def trainable_parameters(self):
        return self.peft.parameters()
