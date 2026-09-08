# ckpt.py - 检查点保存/加载详解

## 文件概述

`ckpt.py` 提供模型检查点的保存和加载功能，支持训练中断恢复和最佳模型保存。

## 核心函数

### save_checkpoint() - 保存检查点

```python
def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    extra: dict[str, Any] | None = None,
) -> None:
    """
    保存训练检查点
    
    参数:
        path: 保存路径
        model: 要保存的模型
        optimizer: 优化器（保存状态）
        epoch: 当前epoch
        best_metric: 最佳指标值
        extra: 额外保存的数据（如任务头参数）
    """
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
```

### load_checkpoint() - 加载检查点

```python
def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """
    加载训练检查点

    参数:
        path: 检查点路径
        model: 要加载参数的模型
        optimizer: 优化器 (可选)
        strict: 是否严格匹配 state_dict keys (默认 True)
                推理时使用 strict=False 以兼容 LUPI 扩展键
        optimizer: 优化器（可选，加载状态）
    
    返回:
        检查点中的额外信息
    """
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    
    return state
```

## 检查点文件结构

```
best.pt 内容:
├── epoch: int                        # 当前epoch
├── model_state_dict: OrderedDict     # 模型参数
├── optimizer_state_dict: dict        # 优化器状态
├── best_metric: float                # 最佳指标值
└── head_state_dict: OrderedDict      # 任务头参数 (extra)

文件大小估算（约2M-10M）:
- MTCNBackbone: ~4-8M (约100万-200万参数)
- GroupedModel: ~1M (约30万参数)
- TaskHead: ~0.1M (约2万参数)
- 总计: ~5-10M
```

## 使用示例

```python
from common.utils.ckpt import save_checkpoint, load_checkpoint
from common.models.mtcn_backbone import MTCNBackbone
from common.models.heads import A1Head
import torch.optim as optim

# 创建模型
model = MTCNBackbone(cfg)
head = A1Head(d_in=256)
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

# 训练时保存最佳模型
if val_metric > best_metric:
    save_checkpoint(
        path=Path("checkpoints/best.pt"),
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        best_metric=val_metric,
        extra={"head_state_dict": head.state_dict()},
    )

# 推理时加载
state = load_checkpoint(Path("checkpoints/best.pt"), model, optimizer=None)
head.load_state_dict(state["head_state_dict"])

# 恢复训练
state = load_checkpoint(Path("checkpoints/best.pt"), model, optimizer)
start_epoch = state["epoch"] + 1
best_metric = state["best_metric"]
```