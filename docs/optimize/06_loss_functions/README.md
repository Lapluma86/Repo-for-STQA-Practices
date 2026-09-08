# 损失函数优化

## 问题诊断（紧急）

当前代码存在**严重的接口不匹配**问题：

### runner.py 调用的参数 vs heads.py 实际支持的参数

**A1 损失函数 (`a1_loss`)**：
```python
# runner.py 实际调用（第 424-430 行）：
main_loss = a1_loss(
    p_logits, targets, pos_weight=pos_weight,
    label_smoothing=label_smoothing,
    use_combined=use_combined_loss,      # ❌ heads.py 不支持
    gamma_neg=gamma_neg,                  # ❌ heads.py 不支持
    gamma_pos=gamma_pos,                  # ❌ heads.py 不支持
    clip=clip,                            # ❌ heads.py 不支持
    soft_f1_weight=soft_f1_weight,        # ❌ heads.py 不支持
)

# heads.py 实际签名：
def a1_loss(logits, targets, pos_weight=None, label_smoothing=0.0):
    ...  # 只支持 4 个参数
```

**A2 损失函数 (`a2_ordinal_loss`)**：
```python
# runner.py 实际调用（第 432-437 行）：
main_loss = a2_ordinal_loss(
    p_logits, targets, pos_weight=pos_weight,
    label_smoothing=label_smoothing,
    use_corn=use_corn_loss,   # ❌ heads.py 不支持
    use_qwk=use_qwk_aux,     # ❌ heads.py 不支持
    qwk_weight=qwk_weight,   # ❌ heads.py 不支持
)

# heads.py 实际签名：
def a2_ordinal_loss(logits, labels, pos_weight=None, label_smoothing=0.0):
    ...  # 只支持 4 个参数
```

**结论**: 配置文件中声明的 ASL、CORN、QWK 等高级损失函数从未被实现。这些正是提升指标的关键手段。

## 改进方案

| 文件 | 方案 | 预期提升 | 优先级 |
|------|------|----------|--------|
| `asymmetric_loss.py` | ASL + Soft-F1 联合损失 | A1 F1 +3~8% | **P0** |
| `ordinal_loss_enhanced.py` | CORN + 可微 QWK 辅助损失 | A2 QWK +3~8% | **P0** |

## 文件说明

### asymmetric_loss.py

解决 A1 类别不平衡问题，提供三层递进的损失函数：

1. **`asymmetric_loss`**: 非对称 Focal Loss
   - `gamma_neg=2`: 强力抑制容易的负样本（正常人群）
   - `gamma_pos=0`: 不抑制正样本（异常人群每个都重要）
   - `clip=0.05`: 极低概率负样本直接截断梯度

2. **`soft_f1_loss`**: 可微 F1 损失
   - 直接优化评价指标 F1 的软版本
   - 解决 "优化 BCE ≠ 优化 F1" 的 surrogate gap

3. **`a1_loss_enhanced`**: 兼容 runner.py 的完整替换函数
   - `use_combined=False` → 退化为原始 BCE（向后兼容）
   - `use_combined=True` → ASL + α×Soft-F1

### ordinal_loss_enhanced.py

解决 A2 序数回归中"训练目标与评价指标不对齐"问题：

1. **`corn_loss`**: 条件序数回归损失
   - 建模条件概率 P(Y≥k | Y≥k-1) → 天然保证单调性
   - 只在"已通过前一阈值"的样本上训练当前阈值

2. **`differentiable_qwk_loss`**: 可微 QWK 损失
   - 构造软混淆矩阵 → 直接优化 QWK
   - QWK 对大错误惩罚更重（(0-3)²=9 vs (2-3)²=1）→ 避免灾难性错误

3. **`a2_ordinal_loss_enhanced`**: 兼容 runner.py 的完整替换函数
   - 基础 BCE + (可选)CORN + (可选)α×QWK

## 集成方式

最简单的集成（直接替换 heads.py 中的函数）：

```python
# 在 heads.py 文件末尾添加或替换：

# 方法1：导入替换
from docs.optimize.loss_functions.asymmetric_loss import a1_loss_enhanced as a1_loss
from docs.optimize.loss_functions.ordinal_loss_enhanced import a2_ordinal_loss_enhanced as a2_ordinal_loss

# 方法2：直接将函数复制到 heads.py 中，替换原有的 a1_loss 和 a2_ordinal_loss
```

## 注意事项

1. `a1_loss_enhanced` 和 `a2_ordinal_loss_enhanced` 都是**向后兼容**的：
   不传新增参数时，行为与原版完全一致
2. `differentiable_qwk_loss` 中的软混淆矩阵计算在 batch 较小时方差大，
   建议 `grad_accumulation ≥ 4` 以获得稳定的梯度信号
3. `soft_f1_loss` 在 batch 维度上聚合，batch 太小会导致 F1 估计不准
