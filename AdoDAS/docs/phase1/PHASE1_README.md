# 第一阶段优化 - README

## 📋 概述

本次优化基于现有项目实现，针对抑郁/焦虑/压力评估任务（A1）和21项心理评估任务（A2）进行了三个方向的优化分析和实施。

---

## ✅ 优化内容

### 1. 类别平衡损失函数（立即可用）

**A1 任务**：ASL + Soft-F1
- 非对称损失（ASL）：对正/负样本使用不同的 focusing 参数
- Soft-F1：直接优化 F1 指标的可微版本
- **预期提升**：F1 +3~8%

**A2 任务**：CORN + QWK
- CORN：条件序数回归，保证单调性
- QWK：直接优化 Cohen's Kappa 指标
- **预期提升**：QWK +3~5%

**状态**：✅ 已在 `common/models/heads.py` 实现，只需修改配置即可启用

### 2. 不确定性加权多任务学习（需要辅助标签）

- 自动平衡多任务损失，避免手动调参
- 新增辅助任务：情绪维度预测、情感分类、面部AU预测
- **预期提升**：主任务指标 +5~10%

**状态**：✅ 已实现，需要辅助标签才能启用

### 3. 池化层优化（暂缓）

- **决策**：保持现有 ASP，暂不改为 Cross-Modal ASP
- **理由**：架构重构风险大，现有性能足够

---

## 🚀 快速开始（3步）

### 步骤 1：测试优化模块
```bash
python test_phase1_optimization.py
```

### 步骤 2：使用优化配置训练
```bash
# A1 任务
python train.py --task a1 --config tasks/a1/phase1_optimization.yaml

# A2 任务
python train.py --task a2 --config tasks/a2/phase1_optimization.yaml
```

### 步骤 3：查看结果
```bash
# 查看训练日志
tail -f output/a1/logs/train_grouped_a1_*.log

# 查看验证集指标
# A1: 关注 mean_f1 和 pcf1
# A2: 关注 mean_qwk 和 mean_mae
```

---

## 📁 文件说明

### 核心代码
- `common/models/mtl_uncertainty.py` - 不确定性加权 + 辅助任务头
- `common/models/phase1_integration.py` - 集成脚本
- `common/models/heads.py` - 损失函数（ASL/CORN/QWK已实现）

### 配置文件
- `tasks/a1/phase1_optimization.yaml` - A1 任务优化配置
- `tasks/a2/phase1_optimization.yaml` - A2 任务优化配置

### 文档
- `docs/PHASE1_OPTIMIZATION_SUMMARY.md` - 快速总结（本文档的详细版）
- `docs/PHASE1_OPTIMIZATION_GUIDE.md` - 详细实施指南

### 测试
- `test_phase1_optimization.py` - 测试脚本

---

## 📊 配置对比

### A1 任务配置变化

| 参数 | default.yaml | phase1_optimization.yaml | 说明 |
|------|--------------|--------------------------|------|
| `use_combined_loss` | - | 1 | 启用 ASL + Soft-F1 |
| `gamma_neg` | - | 2.0 | 负样本 focusing |
| `soft_f1_weight` | - | 0.3 | Soft-F1 权重 |
| `label_smoothing` | 0.0 | 0.05 | 标签平滑 |
| `feature_noise_std` | 0.0 | 0.01 | 特征噪声 |
| `session_drop_prob` | 0.0 | 0.1 | 会话 Dropout |
| `patience` | 6 | 8 | 早停耐心 |

### A2 任务配置变化

| 参数 | default.yaml | phase1_optimization.yaml | 说明 |
|------|--------------|--------------------------|------|
| `use_corn_loss` | 1 | 1 | CORN 损失（保持） |
| `use_qwk_aux` | 1 | 1 | QWK 损失（保持） |
| `qwk_weight` | 0.3 | 0.3 | QWK 权重（保持） |
| `label_smoothing` | 0.05 | 0.05 | 标签平滑（保持） |
| `feature_noise_std` | 0.01 | 0.01 | 特征噪声（保持） |
| `session_drop_prob` | 0.1 | 0.1 | 会话 Dropout（保持） |

---

## 🎯 预期效果

### A1 任务

| 指标 | Baseline | + 优化损失 | 提升 |
|------|----------|-----------|------|
| mean_f1 | 0.65~0.70 | 0.70~0.75 | +3~8% |
| D F1 | - | 提升 | - |
| A F1 | - | 提升 | - |
| S F1 | - | 提升 | - |

### A2 任务

| 指标 | Baseline | + 优化损失 | 提升 |
|------|----------|-----------|------|
| mean_qwk | 0.55~0.60 | 0.58~0.63 | +3~5% |
| mean_mae | - | 降低 | - |

---

## 🔧 调参建议

### A1 任务

如果 F1 提升不明显：
```yaml
gamma_neg: 3.0          # 增大（从 2.0）
soft_f1_weight: 0.5     # 增大（从 0.3）
```

如果训练不稳定：
```yaml
lr: 0.0005              # 降低（从 0.001）
soft_f1_weight: 0.2     # 降低（从 0.3）
```

### A2 任务

如果 QWK 下降：
```yaml
qwk_weight: 0.1         # 降低（从 0.3）
# 或
use_qwk_aux: 0          # 禁用
```

如果过拟合：
```yaml
label_smoothing: 0.1    # 增大（从 0.05）
feature_noise_std: 0.02 # 增大（从 0.01）
dropout: 0.3            # 增大（从 0.2）
```

---

## 📖 详细文档

### 快速参考
- **本文档**：快速开始和配置说明
- `docs/PHASE1_OPTIMIZATION_SUMMARY.md`：完整总结

### 详细指南
- `docs/PHASE1_OPTIMIZATION_GUIDE.md`：
  - 详细实施步骤
  - 配置参数说明
  - 故障排查
  - 启用完整优化（MTL）

---

## 🐛 常见问题

**Q: 训练不稳定，损失震荡**  
A: 降低学习率或辅助损失权重

**Q: A1 F1 提升不明显**  
A: 增大 `gamma_neg` 或 `soft_f1_weight`

**Q: A2 QWK 下降**  
A: 降低 `qwk_weight` 或禁用 `use_qwk_aux`

**Q: 如何启用不确定性加权 MTL？**  
A: 需要准备辅助标签，详见 `docs/PHASE1_OPTIMIZATION_GUIDE.md`

---

## 📚 参考文献

1. Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh Losses", CVPR 2018
2. Ridnik et al., "Asymmetric Loss For Multi-Label Classification", ICCV 2021
3. Shi et al., "CORN: Conditional Ordinal Regression for Neural Networks", Pattern Recognition 2021

---

## 🎓 核心优势

1. **最小侵入**：只需修改配置文件
2. **渐进式**：可逐步启用优化功能
3. **理论支撑**：基于顶会论文，非玄学调参
4. **向后兼容**：可随时回退到原配置

---

## 下一步

1. ✅ 运行测试：`python test_phase1_optimization.py`
2. ✅ 快速训练：`--epochs 5` 验证效果
3. ✅ 查看日志：关注损失和指标
4. ✅ 调整参数：根据验证集表现微调
5. ✅ 完整训练：确认效果后完整训练

---

## 联系

如有问题，请查看：
- 详细指南：`docs/PHASE1_OPTIMIZATION_GUIDE.md`
- 训练日志：`output/logs/train_grouped_*.log`
- 测试脚本：`test_phase1_optimization.py`
