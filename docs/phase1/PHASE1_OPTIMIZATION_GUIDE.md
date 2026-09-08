# 第一阶段优化实施指南

## 优化概览

基于现有项目实现，第一阶段优化包含以下三个方向：

### ✅ 1. 不确定性加权多任务学习（MTL）
- **动态权重**：用可学习参数自动平衡多任务损失，避免手动调参
- **新增辅助任务**：
  - 情绪维度预测（valence/arousal）- 回归任务
  - 语音情感分类（4类基础情绪）- 分类任务
  - 面部动作单元 AU 预测（12个关键AU）- 多标签分类

### ✅ 2. 类别平衡损失函数
- **A1任务**：ASL（非对称损失）+ Soft-F1（已在 `heads.py` 实现）
- **A2任务**：Ordinal BCE + CORN + QWK（已在 `heads.py` 实现）

### ⚠️ 3. 池化层优化（暂缓）
- **决策**：保持现有 ASP，暂不改为 Cross-Modal ASP
- **理由**：架构重构风险大，现有 ASP 已结合 VAD/QC 信号，性能足够

---

## 文件结构

```
AdoDAS2026/
├── common/models/
│   ├── mtl_uncertainty.py          # 新增：不确定性加权 + 辅助任务头
│   ├── phase1_integration.py       # 新增：集成脚本
│   ├── grouped_model.py            # 已有：分组模型
│   └── heads.py                    # 已有：损失函数（ASL/CORN/QWK已实现）
├── tasks/
│   ├── a1/
│   │   ├── default.yaml
│   │   └── phase1_optimization.yaml    # 新增：A1优化配置
│   └── a2/
│       ├── default.yaml
│       └── phase1_optimization.yaml    # 新增：A2优化配置
├── docs/
│   ├── PHASE1_OPTIMIZATION_GUIDE.md    # 本文档
│   └── PHASE1_OPTIMIZATION_SUMMARY.md
└── test_phase1_optimization.py         # 测试脚本
```

---

## 实施步骤

### 步骤 1：验证现有损失函数

当前 `heads.py` 已经实现了优化损失函数，验证是否正常工作：

```python
# 检查 heads.py 第 262-478 行
# - a1_loss 支持 use_combined, gamma_neg, soft_f1_weight 等参数 ✓
# - a2_ordinal_loss 支持 use_corn, use_qwk, qwk_weight 等参数 ✓
```

**测试命令**：
```bash
# 测试优化模块
python test_phase1_optimization.py

# 快速测试 A1 任务的 ASL + Soft-F1（2个epoch）
python train.py --task a1 --config tasks/a1/phase1_optimization.yaml --epochs 2

# 快速测试 A2 任务的 CORN + QWK（2个epoch）
python train.py --task a2 --config tasks/a2/phase1_optimization.yaml --epochs 2
```

### 步骤 2：准备辅助任务标签（可选）

如果你的数据集有以下标注，可以启用对应的辅助任务：

1. **情绪维度标签**（valence/arousal）
   - 格式：`(B, 2)` 浮点数，范围 [-1, 1]
   - 来源：可以从现有的抑郁/焦虑分数推导，或使用预训练模型标注

2. **情感分类标签**（4类）
   - 格式：`(B,)` 整数，0=快乐, 1=悲伤, 2=愤怒, 3=中性
   - 来源：可以从音频/视频特征用预训练模型标注

3. **面部AU标签**（12个AU）
   - 格式：`(B, 12)` 二元标签
   - 来源：OpenFace 已经提取了 AU 特征，可以直接使用

**如果没有这些标签**：
- 配置文件中已默认禁用辅助任务（`enable_auxiliary_tasks: false`）
- 只使用优化损失函数即可获得显著提升

### 步骤 3：使用优化配置训练

```bash
# A1 任务（抑郁/焦虑/压力分类）
python train.py --task a1 --config tasks/a1/phase1_optimization.yaml

# A2 任务（21项评估）
python train.py --task a2 --config tasks/a2/phase1_optimization.yaml
```

---

## 配置参数说明

### A1 任务优化参数

```yaml
# 类别平衡损失（ASL + Soft-F1）
use_combined_loss: 1              # 启用联合损失
gamma_neg: 2.0                    # 负样本 focusing（越大越抑制简单负样本）
gamma_pos: 0.0                    # 正样本 focusing（0=不抑制）
clip: 0.05                        # 概率截断阈值
soft_f1_weight: 0.3               # Soft-F1 权重

# 数据增强
label_smoothing: 0.05             # 从 0.0 增加到 0.05
feature_noise_std: 0.01           # 从 0.0 增加到 0.01
session_drop_prob: 0.1            # 从 0.0 增加到 0.1

# 早停策略
patience: 8                       # 从 6 增加到 8
early_stop_metric: primary        # 使用 F1 作为早停依据
```

**调参建议**：
- `gamma_neg`：类别越不平衡，设置越大（1.5~3.0）
- `clip`：0.05~0.1，过大会丢失梯度
- `soft_f1_weight`：0.2~0.5，过大会不稳定

### A2 任务优化参数

```yaml
# 序数回归增强损失（CORN + QWK）
use_corn_loss: 1                  # 启用条件序数损失
use_qwk_aux: 1                    # 启用 QWK 辅助损失
qwk_weight: 0.3                   # QWK 权重

# 使用 CORAL head
use_coral: true                   # 改进的序数回归头

# 数据增强
label_smoothing: 0.05
feature_noise_std: 0.01
session_drop_prob: 0.1

# 早停策略
patience: 10
early_stop_metric: val_loss       # 使用验证损失作为早停依据
```

**调参建议**：
- `qwk_weight`：0.2~0.5，直接优化评价指标
- 如果验证集 QWK 不稳定，降低 `qwk_weight`

### 不确定性加权 MTL（可选）

```yaml
# 需要辅助标签才能启用
use_uncertainty_weighting: false  # 暂不启用
enable_auxiliary_tasks: false     # 暂不启用
enable_emotion_dims: false        # 情绪维度预测
enable_emotion_cls: false         # 情感分类
enable_au_pred: false             # AU 预测
```

---

## 预期效果

### A1 任务（抑郁/焦虑/压力）
- **Baseline**（BCE + pos_weight）：F1 ≈ 0.65~0.70
- **+ ASL + Soft-F1**：F1 ≈ 0.70~0.75（+3~8%）
- **+ 不确定性加权 MTL**：F1 ≈ 0.72~0.77（+5~10%）

### A2 任务（21项评估）
- **Baseline**（Ordinal BCE）：QWK ≈ 0.55~0.60
- **+ CORN + QWK**：QWK ≈ 0.58~0.63（+3~5%）
- **+ 不确定性加权 MTL**：QWK ≈ 0.60~0.65（+5~8%）

---

## 监控指标

训练时关注以下日志：

### 损失值
```
Epoch 10/30:
  main_loss: 0.4523
  session_loss: 0.3891
  session_type_loss: 0.8234
```

### 验证集指标
- **A1**：关注 `mean_f1` 和 `pcf1`（各类别 F1）
  ```
  Val 10/30: loss=0.4234 mean_f1=0.7234 auroc=0.8123
    D: F1=0.7123  A: F1=0.7234  S: F1=0.7345
  ```

- **A2**：关注 `mean_qwk` 和 `mean_mae`
  ```
  Val 10/40: loss=0.3456 mean_qwk=0.6234 mean_mae=0.4567
    pred dist: 0=25.3% 1=38.2% 2=28.1% 3=8.4%
    GT   dist: 0=23.1% 1=40.5% 2=27.3% 3=9.1%
  ```

---

## 故障排查

### 问题 1：训练不稳定，损失震荡
**原因**：学习率过大或损失权重不合适

**解决**：
```yaml
# 降低学习率
lr: 0.0005  # 从 0.001 降低

# 或降低辅助损失权重
soft_f1_weight: 0.2  # A1: 从 0.3 降低
qwk_weight: 0.2      # A2: 从 0.3 降低
```

### 问题 2：A1 F1 提升不明显
**原因**：类别不平衡不严重，或 ASL 参数不合适

**解决**：
```yaml
# 增大 gamma_neg，更激进地抑制简单负样本
gamma_neg: 3.0  # 从 2.0 增大

# 增大 Soft-F1 权重
soft_f1_weight: 0.5  # 从 0.3 增大

# 检查正样本率
# 如果 > 30%，ASL 效果有限，考虑用标准 BCE
use_combined_loss: 0
```

### 问题 3：A2 QWK 下降
**原因**：QWK 辅助损失权重过大，干扰主任务

**解决**：
```yaml
# 降低 QWK 权重
qwk_weight: 0.1  # 从 0.3 降低

# 或禁用 QWK 辅助损失
use_qwk_aux: 0
```

### 问题 4：过拟合严重
**原因**：数据增强不足或正则化不够

**解决**：
```yaml
# 增加数据增强
label_smoothing: 0.1        # 从 0.05 增大
feature_noise_std: 0.02     # 从 0.01 增大
session_drop_prob: 0.2      # 从 0.1 增大

# 增加正则化
dropout: 0.3                # 从 0.2 增大
weight_decay: 0.02          # 从 0.01 增大
```

---

## 启用完整优化（需要辅助标签）

如果你准备好了辅助任务标签，可以启用不确定性加权 MTL：

### 步骤 1：修改数据集

编辑 `common/data/grouped_dataset.py`，在 `__getitem__` 中添加辅助标签：

```python
def __getitem__(self, idx: int) -> dict:
    # ... 现有代码 ...
    
    # 添加辅助标签（如果有）
    auxiliary_targets = None
    if self.split == "train":  # 只在训练集加载
        auxiliary_targets = {
            "emotion_dims": self._load_emotion_dims(anon_pid),  # (2,) 或 None
            "emotion_cls": self._load_emotion_cls(anon_pid),    # 标量 或 -1
            "au_labels": self._load_au_labels(anon_pid),        # (12,) 或 None
        }
    
    return {
        # ... 现有返回值 ...
        "auxiliary_targets": auxiliary_targets,
    }
```

### 步骤 2：修改配置文件

```yaml
# 启用不确定性加权 MTL
use_uncertainty_weighting: true
enable_auxiliary_tasks: true
enable_emotion_dims: true
enable_emotion_cls: true
enable_au_pred: true
```

### 步骤 3：集成优化模块

在 `runner.py` 中集成 `phase1_integration.py`：

```python
from common.models.phase1_integration import (
    create_optimized_model,
    compute_optimized_loss,
)

# 创建优化模型
optimized_model = create_optimized_model(
    grouped_model=grouped_model,
    participant_head=participant_head,
    session_head=session_head,
    cfg=cfg,
    d_shared=bb_cfg.d_shared,
    aux_dim=aux_dim,
).to(device)

# 训练循环中
outputs = optimized_model(flat_batch, B, session_valid, aux_attrs)
loss, loss_dict = compute_optimized_loss(
    outputs=outputs,
    targets={
        "participant_y": targets,
        "session_types": session_types,
        "auxiliary_targets": batch.get("auxiliary_targets"),
    },
    model=optimized_model,
    task=task,
    session_valid=session_valid,
    pos_weight=pos_weight_t,
    **cfg,  # 传入所有损失参数
)
```

---

## 下一步优化方向

第一阶段完成后，可以考虑：

1. **数据增强**：
   - 时序增强（时间扭曲、裁剪）
   - 模态 Dropout（随机丢弃某个模态）
   - SpecAugment（频谱增强）

2. **模型架构**：
   - Cross-Modal Attention（跨模态交互）
   - Transformer 替换 TCN（如果数据量足够）
   - 多尺度特征融合

3. **训练策略**：
   - EMA（指数移动平均）
   - SWA（随机权重平均）
   - Mixup（样本混合）
   - 对比学习

4. **后处理**：
   - 测试时增强（TTA）
   - 模型集成（Ensemble）
   - 阈值优化

---

## 参考文献

1. **不确定性加权 MTL**：
   Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics", CVPR 2018

2. **非对称损失（ASL）**：
   Ridnik et al., "Asymmetric Loss For Multi-Label Classification", ICCV 2021

3. **CORN 损失**：
   Shi et al., "CORN: Conditional Ordinal Regression for Neural Networks", Pattern Recognition 2021

4. **Soft-F1 损失**：
   直接优化 F1 指标的可微近似

---

## 快速命令参考

```bash
# 测试优化模块
python test_phase1_optimization.py

# A1 任务训练（完整）
python train.py --task a1 --config tasks/a1/phase1_optimization.yaml

# A2 任务训练（完整）
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

## 联系与支持

如有问题，请检查：
1. 日志文件：`output/logs/train_grouped_*.log`
2. 配置文件：`tasks/a1/phase1_optimization.yaml` 或 `tasks/a2/phase1_optimization.yaml`
3. 代码实现：`common/models/phase1_integration.py`
4. 测试脚本：`test_phase1_optimization.py`
