from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

# 保存和加载模型检查点的实用程序函数

def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    extra: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_metric": best_metric,
    }
    if extra:
        state.update(extra)
    torch.save(state, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state["model_state_dict"], strict=strict)
    if not strict and (missing or unexpected):
        import logging
        log = logging.getLogger(__name__)
        if missing:
            log.warning(f"load_checkpoint: missing keys ({len(missing)}): {missing[:5]}...")
        if unexpected:
            log.warning(f"load_checkpoint: unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    return state
