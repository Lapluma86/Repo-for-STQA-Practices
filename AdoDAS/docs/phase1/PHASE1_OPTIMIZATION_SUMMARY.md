# 第一阶段优化总结

## 优化决策

基于你的需求和现有项目实现，我完成了第一阶段优化的分析和实施方案。

---

## ✅ 已实施的优化

### 1. 不确定性加权多任务学习（MTL）

**文件**: `common/models/mtl_uncertainty.py`

**核心功能**:
- `UncertaintyWeightedLoss`: 自动平衡多任务损失，避免手动调参
- `MultiTaskHead`: 辅助任务预测头（情绪维度、情感分类、AU预测）
- `compute_auxiliary_losses`: 辅助任务损失计算

**理论基础**: Kendall et al., CVPR 2018
- 用可学习的不确定性参数 σ 自动调整任务权重
- 公式: L_total = Σ [(1/2σ²)×L_i + log(σ)]

**预期提升**: 主任务指标 +5~10%

---

### 2. 类别平衡损失函数

**状态**: ✅ 已在 `common/models/heads.py` 实现

#### A1 任务（抑郁/焦虑/压力）
- **ASL (Asymmetric Loss)**: 对正/负样本使用不同的 focusing 参数
  - `gamma_neg=2.0`: 抑制简单负样本
  - `gamma_pos=0.0`: 不抑制稀有正样本
  - `clip=0.05`: 硬截断极低概率负样本
  
- **Soft-F1 Loss**: 直接优化 F1 指标的可微版本
  - 解决 "优化 BCE ≠ 优化 F1" 的问题

**实现位置**: `heads.py` 第 262-306 行
**预期提升**: F1 +3~8%

#### A2 任务（21项评估）
- **CORN Loss**: 条件序数回归，保证单调性
- **QWK Loss**: 直接优化 Cohen's Kappa 指标

**实现位置**: `heads.py` 第 308-478 行
**预期提升**: QWK +3~5%

---

### 3. 集成脚本

**文件**: `common/models/phase1_integration.py`

**核心功能**:
- `OptimizedGroupedModel`: 封装优化后的完整模型
- `compute_optimized_loss`: 统一的损失计算接口
- `create_optimized_model`: 模型创建工厂函数

**优势**: 
- 最小化对现有代码的侵入
- 提供向后兼容的接口
- 支持渐进式启用优化功能

---

## ⚠️ 暂缓的优化

### 池化层优化（ASP → Cross-Modal ASP）

**决策**: 暂不实施

**理由**:
1. 现有 ASP 已经结合了 VAD 和 QC 信号，性能足够
2. 你的架构是先各模态独立池化再融合，这是合理的设计
3. 改为 Cross-Modal ASP 需要重构整个池化逻辑，风险大

**替代方案**:
- 如果后续需要跨模态交互，在 TCN 之后、池化之前加 Cross-Modal Attention 更安全
- 或者在参与者级聚合时引入跨会话注意力

---

## 📁 新增文件

```
AdoDAS2026/
├── common/models/
│   ├── mtl_uncertainty.py           # 不确定性加权 + 辅助任务头
│   └── phase1_integration.py        # 集成脚本
├── tasks/
│   ├── a1/
│   │   ├── default.yaml
│   │   └── phase1_optimization.yaml # A1 优化配置
│   └── a2/
│       ├── default.yaml
│       └── phase1_optimization.yaml # A2 优化配置
├── docs/
│   ├── PHASE1_OPTIMIZATION_GUIDE.md    # 详细实施指南
│   └── PHASE1_OPTIMIZATION_SUMMARY.md  # 本文档
└── test_phase1_optimization.py         # 测试脚本
```

---

## 🚀 快速开始

### 步骤 1: 测试优化模块

```bash
python test_phase1_optimization.py
```

### 步骤 2: 使用优化配置训练

```bash
# A1 任务（抑郁/焦虑/压力分类）
python train.py --task a1 --config tasks/a1/phase1_optimization.yaml

# A2 任务（21项评估）
python train.py --task a2 --config tasks/a2/phase1_optimization.yaml

# 快速测试（2个epoch）
python train.py --task a1 --config tasks/a1/phase1_optimization.yaml --epochs 2
```

这会启用：
- ✅ ASL + Soft-F1（A1）
- ✅ CORN + QWK（A2）
- ✅ 标签平滑
- ✅ 特征噪声增强
- ✅ 会话级 Dropout

### 步骤 3: 完整优化（需要辅助标签）

如果你有情绪维度、情感分类、AU 等辅助标签，可以启用完整优化：

1. 修改 `common/data/grouped_dataset.py`，添加辅助标签加载
2. 在配置文件中启用：
   ```yaml
   use_uncertainty_weighting: true
   enable_auxiliary_tasks: true
   ```
3. 在 `runner.py` 中集成 `phase1_integration.py`

详见 `docs/PHASE1_OPTIMIZATION_GUIDE.md`

---

## 📊 预期效果

### A1 任务（抑郁/焦虑/压力分类）

| 方法 | F1 Score | 提升 |
|------|----------|------|
| Baseline (BCE + pos_weight) | 0.65~0.70 | - |
| + ASL + Soft-F1 | 0.70~0.75 | +3~8% |
| + 不确定性加权 MTL | 0.72~0.77 | +5~10% |

### A2 任务（21项评估）

| 方法 | QWK | 提升 |
|------|-----|------|
| Baseline (Ordinal BCE) | 0.55~0.60 | - |
| + CORN + QWK | 0.58~0.63 | +3~5% |
| + 不确定性加权 MTL | 0.60~0.65 | +5~8% |

---

## 🔧 关键配置参数

### A1 任务（tasks/a1/phase1_optimization.yaml）

```yaml
# ASL + Soft-F1 损失
use_combined_loss: 1
gamma_neg: 2.0        # 负样本 focusing（1.5~3.0）
gamma_pos: 0.0        # 正样本 focusing（通常为0）
clip: 0.05            # 概率截断（0.05~0.1）
soft_f1_weight: 0.3   # Soft-F1 权重（0.2~0.5）

# 数据增强
label_smoothing: 0.05
feature_noise_std: 0.01
session_drop_prob: 0.1

# 早停策略
patience: 8
early_stop_metric: primary  # 使用 F1 作为早停依据
```

### A2 任务（tasks/a2/phase1_optimization.yaml）

```yaml
# CORN + QWK 损失
use_corn_loss: 1
use_qwk_aux: 1
qwk_weight: 0.3       # QWK 权重（0.2~0.5）

# 使用 CORAL head
use_coral: true

# 数据增强
label_smoothing: 0.05
feature_noise_std: 0.01
session_drop_prob: 0.1

# 早停策略
patience: 10
early_stop_metric: val_loss  # 使用验证损失作为早停依据
```

### 不确定性加权 MTL（可选）

```yaml
# 需要辅助标签才能启用
use_uncertainty_weighting: false  # 默认禁用
enable_auxiliary_tasks: false     # 默认禁用
enable_emotion_dims: false        # 情绪维度预测
enable_emotion_cls: false         # 情感分类
enable_au_pred: false             # AU 预测
```

---

## 🎯 实施优先级

### P0 - 立即可用（无需修改代码）
✅ ASL + Soft-F1 损失（A1）  
✅ CORN + QWK 损失（A2）  
✅ 标签平滑  
✅ 特征噪声增强  
✅ 会话级 Dropout  

**操作**: 直接使用优化配置训练
```bash
python train.py --task a1 --config tasks/a1/phase1_optimization.yaml
python train.py --task a2 --config tasks/a2/phase1_optimization.yaml
```

### P1 - 需要少量修改（如果有辅助标签）
⚠️ 不确定性加权 MTL  
⚠️ 辅助任务（情绪维度、情感分类、AU）  

**操作**: 
1. 修改数据集加载辅助标签
2. 集成 `phase1_integration.py`
3. 启用配置中的 MTL 参数

### P2 - 后续优化方向
🔜 Cross-Modal Attention  
🔜 数据增强（时序、模态 Dropout）  
🔜 训练策略（EMA、SWA、Mixup）  

---

## 📖 详细文档

- **实施指南**: `docs/PHASE1_OPTIMIZATION_GUIDE.md`
  - 详细步骤
  - 配置说明
  - 故障排查
  - 调参建议

- **测试脚本**: `test_phase1_optimization.py`
  - 验证所有优化模块
  - 确保代码正确性

- **配置文件**: 
  - `tasks/a1/phase1_optimization.yaml`（A1 任务）
  - `tasks/a2/phase1_optimization.yaml`（A2 任务）

---

## ⚡ 核心优势

1. **最小侵入**: 优化损失函数已在 `heads.py` 实现，只需修改配置即可启用
2. **渐进式**: 可以逐步启用优化功能，降低风险
3. **理论支撑**: 所有优化都有顶会论文支持，不是玄学调参
4. **向后兼容**: 保留原有接口，可随时回退

---

## 🔍 监控指标

训练时关注：

### 损失分解
```
Epoch 10/30:
  main_loss: 0.4523
  session_loss: 0.3891
  session_type_loss: 0.8234
```

### 验证集指标
- **A1**: `mean_f1`, `pcf1` (各类别 F1)
  ```
  Val 10/30: mean_f1=0.7234 auroc=0.8123
    D: F1=0.7123  A: F1=0.7234  S: F1=0.7345
  ```

- **A2**: `mean_qwk`, `mean_mae`
  ```
  Val 10/40: mean_qwk=0.6234 mean_mae=0.4567
    pred dist: 0=25.3% 1=38.2% 2=28.1% 3=8.4%
  ```

### 不确定性权重（如果启用 MTL）
```
task_0_weight: 1.23  task_0_sigma: 0.91   # 主任务
task_1_weight: 0.87  task_1_sigma: 1.07   # 会话任务
```
- `weight` 越大 → 任务越重要
- `sigma` 越大 → 不确定性越高 → 权重自动降低

---

## 🐛 常见问题

### Q1: 训练不稳定，损失震荡
**A**: 降低学习率 `lr: 0.0005`，或降低辅助损失权重

### Q2: A1 F1 提升不明显
**A**: 增大 `gamma_neg: 3.0` 或 `soft_f1_weight: 0.5`

### Q3: A2 QWK 下降
**A**: 降低 `qwk_weight: 0.1` 或禁用 `use_qwk_aux: 0`

### Q4: 过拟合严重
**A**: 增加数据增强和正则化
```yaml
label_smoothing: 0.1
feature_noise_std: 0.02
session_drop_prob: 0.2
dropout: 0.3
weight_decay: 0.02
```

详见 `docs/PHASE1_OPTIMIZATION_GUIDE.md` 故障排查章节

---

## 📚 参考文献

1. **Uncertainty Weighting**: Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh Losses", CVPR 2018
2. **Asymmetric Loss**: Ridnik et al., "Asymmetric Loss For Multi-Label Classification", ICCV 2021
3. **CORN Loss**: Shi et al., "CORN: Conditional Ordinal Regression for Neural Networks", Pattern Recognition 2021
4. **Soft-F1**: 直接优化 F1 指标的可微近似

---

## 快速命令参考

```bash
# 测试优化模块
python test_phase1_optimization.py

# A1 任务训练
python train.py --task a1 --config tasks/a1/phase1_optimization.yaml

# A2 任务训练
python train.py --task a2 --config tasks/a2/phase1_optimization.yaml

# 快速测试（2个epoch）
python train.py --task a1 --config tasks/a1/phase1_optimization.yaml --epochs 2

# 覆盖特定参数
python train.py --task a1 --config tasks/a1/phase1_optimization.yaml \
    --gamma_neg 3.0 --soft_f1_weight 0.5

# 查看日志
tail -f output/a1/logs/train_grouped_a1_*.log
```

---

## 下一步

1. **运行测试**: `python test_phase1_optimization.py`
2. **快速训练**: `python train.py --task a1 --config tasks/a1/phase1_optimization.yaml --epochs 5`
3. **查看日志**: 关注损失值和验证集指标
4. **调整参数**: 根据验证集表现微调 `gamma_neg`, `soft_f1_weight` 等
5. **完整训练**: 确认效果后进行完整训练

如有问题，参考 `docs/PHASE1_OPTIMIZATION_GUIDE.md` 或检查日志文件。
