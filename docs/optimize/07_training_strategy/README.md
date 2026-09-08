# 训练策略优化

## 问题诊断

分析训练流程（`runner.py`）发现以下正则化和优化不足：

### 1. 数据增强不足

当前训练仅有两种正则化手段：
- `feature_noise_std=0.01`（微弱的高斯噪声）
- `session_drop_prob=0.1`（偶尔丢弃一个会话）

对于 ~1000 样本的小数据集，这远远不够。过拟合是当前最大的性能瓶颈。

### 2. 权重选择过于脆弱

只保存验证集最佳 epoch 的单个 checkpoint。问题：
- 不同 seed 下最佳 epoch 的 QWK 可能差 5%+
- 单个 epoch 的权重可能落在 loss landscape 的尖锐最小值

### 3. 梯度信号方差大

单次 Dropout 采样 + 小 batch（64）→ 每步梯度的方差大 → 训练不稳定。

## 改进方案

| 文件 | 方案 | 预期提升 | 优先级 | 开销 |
|------|------|----------|--------|------|
| `mixup.py` | 特征空间 Mixup | QWK/F1 +2~5% | **P0** | 几乎无 |
| `multi_sample_dropout.py` | 多采样 Dropout | QWK/F1 +1~2% | P1 | 计算 ×K |
| `swa.py` | 随机权重平均 | QWK +1~3% | P1 | 存储 ×2 |
| `ema.py` | 指数移动平均 | QWK +1~2% | P1 | 存储 ×2 |

## 文件说明

### mixup.py — Mixup 数据增强

最高性价比的正则化手段，尤其适合小数据集。

- **ParticipantMixup**: 在 participant_repr 层面混合两个参与者
  ```
  repr_mix = λ × repr_i + (1-λ) × repr_j
  label_mix = λ × label_i + (1-λ) × label_j
  ```
  λ ~ Beta(0.2, 0.2)，推荐 α=0.2 起步

- **SessionMixup**: 在 session_reprs 层面混合（更细粒度）
  - `intra_participant=True`: 只在同一参与者的不同会话间混合
  - `intra_participant=False`: 跨参与者混合

### multi_sample_dropout.py — 多采样 Dropout

低成本的 ensemble 效果。

- **MultiSampleDropoutHead**: 包装 task_head，K 次 dropout → 平均 logits
- **MultiSampleDropoutLoss**: 包装损失函数，K 次 dropout → 平均 loss（更优）
- K=5 时等效于 5× batch size 的梯度估计精度

### swa.py — 随机权重平均

Epoch 级别的权重平均，找到 loss landscape 的平坦区域。

- **SWAWrapper**: 非侵入式 SWA，只需在训练循环加两行代码
  ```python
  swa.update(epoch)   # 每 epoch 调用
  swa.apply()         # 训练结束后应用
  ```
- 推荐从总 epoch 的 75% 开始累积
- 当前模型用 LayerNorm，不需要更新 BN 统计量

### ema.py — 指数移动平均

Step 级别的平滑，与 SWA 互补。

- **EMA**: 每个训练步更新 `θ_ema = β × θ_ema + (1-β) × θ`
  - β=0.999 → ~1000 步平均窗口
  - 内置 warmup（前 100 步用较小的 decay）
- 提供上下文管理器，验证时临时切换到 EMA 权重：
  ```python
  with ema.apply_temporary():
      val_metrics = validate(model, val_loader)
  ```

## 推荐组合策略

### 最小改动（P0，预期 +5~10%）

1. 加入 Mixup（alpha=0.2）— 3行代码
2. 替换损失函数（使用 06_loss_functions 的增强版）

### 中等改动（P0+P1，预期 +8~15%）

在上述基础上：
3. 加入 Multi-Sample Dropout（K=5）
4. 加入 EMA（decay=0.999）

### 完整方案（全部，预期 +12~20%）

在上述基础上：
5. 加入 SWA（最后 25% epoch）
6. 使用 04_utils/ckpt_optimized.py 的 Top-K 检查点管理

## 注意事项

1. **Mixup + 序数回归**: A2 任务中 Mixup 会产生连续标签（如 1.7），需要对
   序数目标也做插值：`target_mix = λ × [1,1,0] + (1-λ) × [1,0,0] = [1, λ, 0]`
2. **EMA vs SWA**: 两者可以叠加使用。EMA 在训练中提供平滑验证指标，
   SWA 在训练结束后提供最终权重
3. **Multi-Sample Dropout**: K=5 会使 task_head 的前向传播计算量 ×5，
   但 task_head 只是一个线性层（参数量极小），实际开销可忽略
4. **显存考虑**: EMA 需要额外一份权重的显存。模型约 30M 参数 → ~120MB。
   如果显存紧张，优先用 SWA（不额外占显存，只在 CPU 上存平均权重）
