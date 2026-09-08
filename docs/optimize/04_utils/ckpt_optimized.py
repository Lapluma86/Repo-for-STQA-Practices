"""
ckpt_optimized.py — 增强的检查点管理模块

相比原始 ckpt.py 的改进：
1. [P1] save/load 支持 scheduler 和 GradScaler 状态
2. [P1] CheckpointManager：自动保留 Top-K 最优检查点

可直接替换 common/utils/ckpt.py，接口向后兼容。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ============================================================
# 增强的 save/load（向后兼容）
# ============================================================

def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """
    保存检查点，支持 scheduler 和 GradScaler。

    新增参数（均可选，不传则行为与原版一致）：
        scheduler: 学习率调度器，保存其 state_dict 以便断点续训时恢复 LR 曲线
        scaler: torch.amp.GradScaler，混合精度训练时保存缩放状态

    为什么需要保存 scheduler：
        如果使用 CosineAnnealingLR 等调度策略，断点续训时不恢复 scheduler 状态，
        LR 会从初始值重新开始，导致收敛曲线断裂。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_metric": best_metric,
    }
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        state["scaler_state_dict"] = scaler.state_dict()
    if extra:
        state.update(extra)
    torch.save(state, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
) -> dict[str, Any]:
    """
    加载检查点，支持恢复 scheduler 和 GradScaler。

    新增参数（均可选）：
        scheduler: 传入则恢复学习率调度器状态
        scaler: 传入则恢复 GradScaler 状态

    安全性：如果检查点中没有对应的 state_dict 键，会跳过恢复并打印警告，
    不会报错。这保证了新代码能加载旧检查点。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])

    if scheduler is not None:
        if "scheduler_state_dict" in state:
            scheduler.load_state_dict(state["scheduler_state_dict"])
        else:
            logger.warning("检查点中无 scheduler 状态，跳过恢复")

    if scaler is not None:
        if "scaler_state_dict" in state:
            scaler.load_state_dict(state["scaler_state_dict"])
        else:
            logger.warning("检查点中无 scaler 状态，跳过恢复")

    return state


# ============================================================
# 新增：Top-K 检查点管理器
# ============================================================

class CheckpointManager:
    """
    自动保留得分最高的 K 个检查点，淘汰较差的。

    使用场景：
    - 超参搜索时大量检查点占磁盘 → 自动清理
    - 不确定哪个 epoch 最好 → 保留 top-3 而非只保留 best

    工作原理：
    - 每次调用 save_if_top_k 时，比较当前指标与已保存的最差记录
    - 如果当前更优（或记录数 < top_k），保存新检查点
    - 如果超出 top_k 容量，删除最差的检查点文件

    Args:
        save_dir: 检查点保存目录
        top_k: 保留最优的 K 个检查点，默认 3
        mode: "max" 表示指标越大越好（如 F1, QWK），
              "min" 表示指标越小越好（如 MAE, loss）

    用法：
        mgr = CheckpointManager("runs/exp1/checkpoints", top_k=3, mode="max")
        for epoch in range(100):
            val_qwk = evaluate(model)
            saved = mgr.save_if_top_k(
                metric=val_qwk, model=model, optimizer=optimizer,
                epoch=epoch, scheduler=scheduler
            )
            if saved:
                print(f"Epoch {epoch} 进入 top-{mgr.top_k}")
    """

    def __init__(
        self,
        save_dir: str | Path,
        top_k: int = 3,
        mode: str = "max",
    ) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.top_k = top_k
        if mode not in ("max", "min"):
            raise ValueError(f"mode 必须是 'max' 或 'min'，收到 '{mode}'")
        self.mode = mode
        # 记录：(metric_value, file_path)
        self._records: list[tuple[float, Path]] = []

    def save_if_top_k(
        self,
        metric: float,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        scheduler: Any | None = None,
        scaler: Any | None = None,
        extra: dict[str, Any] | None = None,
    ) -> bool:
        """
        如果当前指标进入 top-K，保存检查点并返回 True；否则返回 False。

        自动淘汰超出 top_k 的最差检查点文件。
        """
        should_save = len(self._records) < self.top_k
        if not should_save:
            worst_metric = self._get_worst_metric()
            if self.mode == "max":
                should_save = metric > worst_metric
            else:
                should_save = metric < worst_metric

        if not should_save:
            return False

        # 构建文件名：包含 epoch 和指标值，便于人工识别
        path = self.save_dir / f"ckpt_epoch{epoch}_{metric:.4f}.pt"
        save_checkpoint(
            path=path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_metric=metric,
            scheduler=scheduler,
            scaler=scaler,
            extra=extra,
        )
        self._records.append((metric, path))
        logger.info(f"保存检查点: {path.name} (metric={metric:.4f})")

        # 淘汰超出容量的最差记录
        self._evict()
        return True

    @property
    def best_path(self) -> Path | None:
        """返回当前最优检查点的路径。"""
        if not self._records:
            return None
        if self.mode == "max":
            return max(self._records, key=lambda r: r[0])[1]
        return min(self._records, key=lambda r: r[0])[1]

    @property
    def best_metric(self) -> float | None:
        """返回当前最优指标值。"""
        if not self._records:
            return None
        if self.mode == "max":
            return max(r[0] for r in self._records)
        return min(r[0] for r in self._records)

    def _get_worst_metric(self) -> float:
        """返回当前最差的指标值。"""
        if self.mode == "max":
            return min(r[0] for r in self._records)
        return max(r[0] for r in self._records)

    def _evict(self) -> None:
        """如果记录数超过 top_k，删除最差的。"""
        while len(self._records) > self.top_k:
            # 找到最差记录
            if self.mode == "max":
                worst_idx = min(range(len(self._records)), key=lambda i: self._records[i][0])
            else:
                worst_idx = max(range(len(self._records)), key=lambda i: self._records[i][0])

            _, worst_path = self._records.pop(worst_idx)
            if worst_path.exists():
                worst_path.unlink()
                logger.info(f"淘汰检查点: {worst_path.name}")
