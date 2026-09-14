# SteelRailWay 缺陷修复报告

日期：2026-09-11

---

## 已修复的缺陷

### DEF-004: patch_stride=0 未验证

**文件**: `datasets/rail_dataset.py`

**问题**: `patch_stride` 参数未验证，可能导致无限循环。

**修复**:
```python
if not isinstance(patch_stride, int) or patch_stride <= 0:
    raise ValueError(f"patch_stride must be a positive integer (>0), got {patch_stride}")
```

**额外保护**:
```python
if self.patch_size > self.img_height:
    raise ValueError(f"patch_size cannot be larger than image height")
if self.num_patches <= 0:
    raise ValueError(f"Invalid patch configuration")
```

---

### DEF-001: 输入参数缺乏验证

**文件**: `datasets/rail_dataset.py`

**问题**: 多个参数缺乏输入验证。

**修复**: 在 `__init__` 中为所有参数添加验证：

| 参数 | 验证规则 |
|-----------|------------|
| `view_id` | 1-8 范围 |
| `split` | "train" \| "val" \| "test" |
| `img_size` | 正整数 |
| `depth_norm` | "zscore" \| "minmax" \| "log" |
| `patch_size` | 正整数 |
| `train_sample_ratio` | 0-1 范围 |
| `sampling_mode` | "random" \| "uniform_time" |

同时为 train/val 目录添加了路径验证。

---

### DEF-WB-001: 空列表处理

**文件**: `utils/losses.py`

**问题**: 损失函数对空列表返回 0.0，但未显式处理。

**修复**: 为所有损失函数添加空列表检查：
```python
if not feature_s or not feature_t:
    return torch.tensor(0.0, dtype=torch.float32)
```

受影响的函数：
- `loss_l2()`
- `loss_distil()`
- `loss_distil_p()`
- `loss_distil_pixel()`

---

### DEF-WB-003: 除零保护

**文件**: `eval/metrics_engineering.py`

**修复**: 在 `count_trainable_params()` 中加强保护：
```python
ratio = trainable / total if total > 0 else 0.0
```

---

## 总结

| 缺陷 ID | 优先级 | 修改的文件 |
|-----------|---------|--------------|
| DEF-004 | 高 | `datasets/rail_dataset.py` |
| DEF-001 | 中 | `datasets/rail_dataset.py` |
| DEF-003 | 中 | `datasets/rail_dataset.py` |
| DEF-WB-001 | 低 | `utils/losses.py` |
| DEF-WB-003 | 中 | `eval/metrics_engineering.py` |

---

## 验证示例

修复后，无效输入会抛出异常：

```python
# 无效（抛出 ValueError）
RailDualModalDataset(..., view_id=0, ...)
RailDualModalDataset(..., patch_stride=0, ...)
RailDualModalDataset(..., depth_norm="unknown", ...)

# 有效
RailDualModalDataset(
    train_root="data_20260327",
    test_root="rail_mvtec_gt_test",
    view_id=1,
    split="train",
    img_size=256,
    patch_stride=850,
    depth_norm="zscore"
)
```
