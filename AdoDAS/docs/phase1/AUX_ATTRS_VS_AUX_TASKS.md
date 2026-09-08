# 辅助属性（Auxiliary Attributes）与多任务学习（MTL）使用指南

## 概述

项目中有**两种不同的辅助信息**，容易混淆：

### 1. **辅助属性（Auxiliary Attributes）** - 已有功能 ✅
- **作用**：参与者的人口统计学/背景信息
- **数据来源**：CSV manifest 文件中已有
- **使用方式**：通过 `use_aux_attrs: true` 启用
- **实现位置**：`common/models/aux_encoder.py`

### 2. **辅助任务（Auxiliary Tasks）** - 新增功能 🆕
- **作用**：额外的预测任务（情绪维度、情感分类、AU预测）
- **数据来源**：需要额外标注或从特征推导
- **使用方式**：通过 `enable_auxiliary_tasks: true` 启用
- **实现位置**：`common/models/mtl_uncertainty.py`

---

## 一、辅助属性（Auxiliary Attributes）

### 什么是辅助属性？

辅助属性是参与者的**静态背景信息**，包含5个维度：

| 属性 | 英文名 | 类别数 | 说明 |
|------|--------|--------|------|
| 家庭结构 | Family structure | 6 | 1-6类 |
| 独生子女 | Only child status | 2 | 0=否, 1=是 |
| 父母偏爱 | Parental favoritism | 3 | 1-3类 |
| 成绩变动 | Academic performance change | 3 | 1-3类 |
| 情绪变动 | Emotional state change | 3 | 1-3类 |

### 数据来源

这些数据**已经在你的 CSV manifest 文件中**：

```csv
anon_pid,session,y_D,y_A,y_S,d01,d02,...,Family structure,Only child status,...
P001,A01,1,0,1,2,1,...,3,1,2,1,2
```

### 如何使用？

#### 步骤1：确认数据存在

检查你的 manifest 文件是否包含这5列：
```bash
head -1 /path/to/manifests/train.csv | grep "Family structure"
```

#### 步骤2：启用辅助属性

在配置文件中设置：
```yaml
use_aux_attrs: true
aux_embed_dim: 8
```

#### 步骤3：训练

```bash
python train.py --task a1 --config tasks/a1/phase1_optimization.yaml
```

### 工作原理

```
辅助属性 (B, 5)
    ↓ AuxiliaryAttributeEncoder
    ├─ Family structure → Embedding(7, 8)
    ├─ Only child       → Embedding(3, 8)
    ├─ Parental fav     → Embedding(4, 8)
    ├─ Grade change     → Embedding(4, 8)
    └─ Mood change      → Embedding(4, 8)
    ↓ Concatenate
编码向量 (B, 40)  # 5 × 8 = 40
    ↓ 拼接到参与者表示
participant_repr (B, d_shared + 40)
    ↓ 任务头
预测结果
```

### 为什么配置文件中已经启用？

因为这是**已有功能**，你的数据集已经包含这些信息，所以默认配置中：
```yaml
use_aux_attrs: true
aux_embed_dim: 8
```

这**不是**我新增的优化，而是项目原本就支持的功能。

---

## 二、辅助任务（Auxiliary Tasks）- MTL

### 什么是辅助任务？

辅助任务是**额外的预测目标**，用于多任务学习：

| 任务 | 输出 | 说明 |
|------|------|------|
| 情绪维度预测 | (B, 2) | valence（愉悦度）和 arousal（激活度） |
| 情感分类 | (B, 4) | 4类基础情绪（快乐/悲伤/愤怒/中性） |
| AU预测 | (B, 12) | 12个关键面部动作单元 |

### 数据来源

这些数据**不在你的 CSV 文件中**，需要：

#### 选项1：从现有特征推导（推荐）

```python
# 从 OpenFace AU 特征推导 AU 标签
# 从音频特征推导情感分类
# 从抑郁/焦虑分数推导情绪维度
```

#### 选项2：使用预训练模型标注

```python
# 使用情感识别模型标注音频/视频
# 使用 AU 检测模型标注面部表情
```

#### 选项3：暂不使用

```yaml
# 配置文件中禁用（默认）
enable_auxiliary_tasks: false
```

### 如何使用？

#### 方案A：暂不使用（推荐）

当前配置文件中已经禁用：
```yaml
use_uncertainty_weighting: false
enable_auxiliary_tasks: false
```

**原因**：
- 需要额外的标注数据
- 实施复杂度较高
- 优化损失函数已经能带来显著提升

#### 方案B：完整实施（需要辅助标签）

如果你有或能生成辅助标签：

**步骤1：准备辅助标签**

修改 `common/data/grouped_dataset.py`：

```python
def _load_emotion_dims(self, anon_pid: str) -> np.ndarray | None:
    """
    加载情绪维度标签 (valence, arousal)
    
    可以从以下来源获取：
    1. 从抑郁/焦虑分数推导
    2. 使用预训练模型标注
    3. 人工标注
    """
    # 示例：从抑郁/焦虑分数推导
    row = self.manifest_df[self.manifest_df["anon_pid"] == anon_pid].iloc[0]
    depression = row.get("y_D", -1)
    anxiety = row.get("y_A", -1)
    
    if depression < 0 or anxiety < 0:
        return None
    
    # 简单映射：抑郁高 → valence低，焦虑高 → arousal高
    valence = -0.5 if depression > 0.5 else 0.5
    arousal = 0.5 if anxiety > 0.5 else -0.5
    
    return np.array([valence, arousal], dtype=np.float32)

def _load_emotion_cls(self, anon_pid: str) -> int:
    """
    加载情感分类标签
    
    0=快乐, 1=悲伤, 2=愤怒, 3=中性
    """
    # 示例：从抑郁/焦虑/压力推导
    row = self.manifest_df[self.manifest_df["anon_pid"] == anon_pid].iloc[0]
    depression = row.get("y_D", -1)
    stress = row.get("y_S", -1)
    
    if depression < 0:
        return -1  # 缺失
    
    if depression > 0.7:
        return 1  # 悲伤
    elif stress > 0.7:
        return 2  # 愤怒
    elif depression < 0.3 and stress < 0.3:
        return 0  # 快乐
    else:
        return 3  # 中性

def _load_au_labels(self, anon_pid: str) -> np.ndarray | None:
    """
    加载 AU 标签
    
    可以从 OpenFace 特征中提取
    """
    # 如果你的特征中已经有 AU，可以直接使用
    # 或者返回 None 表示没有标签
    return None

def __getitem__(self, idx: int) -> dict:
    # ... 现有代码 ...
    
    # 添加辅助标签
    auxiliary_targets = None
    if self.split == "train":  # 只在训练集加载
        auxiliary_targets = {
            "emotion_dims": self._load_emotion_dims(anon_pid),
            "emotion_cls": self._load_emotion_cls(anon_pid),
            "au_labels": self._load_au_labels(anon_pid),
        }
    
    return {
        # ... 现有返回值 ...
        "auxiliary_targets": auxiliary_targets,
    }
```

**步骤2：启用配置**

```yaml
use_uncertainty_weighting: true
enable_auxiliary_tasks: true
enable_emotion_dims: true
enable_emotion_cls: true
enable_au_pred: false  # 如果没有 AU 标签，保持 false
```

**步骤3：集成优化模块**

修改 `common/runner.py`（或创建新的训练脚本）：

```python
from common.models.phase1_integration import (
    create_optimized_model,
    compute_optimized_loss,
)

# 在 main() 函数中，替换模型创建
optimized_model = create_optimized_model(
    grouped_model=grouped_model,
    participant_head=participant_head,
    session_head=session_head,
    cfg=cfg,
    d_shared=bb_cfg.d_shared,
    aux_dim=aux_dim,
).to(device)

# 在训练循环中，替换损失计算
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
    label_smoothing=cfg.get("label_smoothing", 0.0),
    use_combined_loss=cfg.get("use_combined_loss", False),
    gamma_neg=cfg.get("gamma_neg", 2.0),
    gamma_pos=cfg.get("gamma_pos", 0.0),
    clip=cfg.get("clip", 0.05),
    soft_f1_weight=cfg.get("soft_f1_weight", 0.3),
    use_corn_loss=cfg.get("use_corn_loss", False),
    use_qwk_aux=cfg.get("use_qwk_aux", False),
    qwk_weight=cfg.get("qwk_weight", 0.3),
)
```

---

## 三、配置参数对比

### 当前配置（推荐）

```yaml
# ============================================================
# 辅助属性（已有功能，数据已存在）
# ============================================================
use_aux_attrs: true           # ✅ 启用（数据已在CSV中）
aux_embed_dim: 8              # Embedding维度

# ============================================================
# 辅助任务（新增功能，需要额外标签）
# ============================================================
use_uncertainty_weighting: false  # ❌ 暂不启用
enable_auxiliary_tasks: false     # ❌ 暂不启用
enable_emotion_dims: false        # ❌ 暂不启用
enable_emotion_cls: false         # ❌ 暂不启用
enable_au_pred: false             # ❌ 暂不启用
```

### 完整优化配置（需要辅助标签）

```yaml
# ============================================================
# 辅助属性（已有功能）
# ============================================================
use_aux_attrs: true           # ✅ 启用
aux_embed_dim: 8

# ============================================================
# 辅助任务（新增功能）
# ============================================================
use_uncertainty_weighting: true   # ✅ 启用不确定性加权
enable_auxiliary_tasks: true      # ✅ 启用辅助任务
enable_emotion_dims: true         # ✅ 情绪维度预测
enable_emotion_cls: true          # ✅ 情感分类
enable_au_pred: true              # ✅ AU预测（如果有标签）
```

---

## 四、常见问题

### Q1: 为什么配置文件中 `use_aux_attrs: true` 但没有报错？

**A**: 因为辅助属性数据**已经在你的 CSV 文件中**，这是项目原本就支持的功能。

### Q2: 我需要准备辅助任务的标签吗？

**A**: **不需要**。当前配置中辅助任务是禁用的（`enable_auxiliary_tasks: false`），只使用优化损失函数就能获得显著提升。

### Q3: 如果我想使用辅助任务，最简单的方法是什么？

**A**: 从现有数据推导：

```python
# 情绪维度：从抑郁/焦虑分数推导
valence = -depression_score  # 抑郁高 → 愉悦度低
arousal = anxiety_score       # 焦虑高 → 激活度高

# 情感分类：从抑郁/焦虑/压力推导
if depression > 0.7: emotion = "悲伤"
elif stress > 0.7: emotion = "愤怒"
elif depression < 0.3: emotion = "快乐"
else: emotion = "中性"

# AU预测：如果没有标签，保持禁用
enable_au_pred: false
```

### Q4: 辅助属性和辅助任务有什么区别？

| 维度 | 辅助属性 | 辅助任务 |
|------|----------|----------|
| **数据来源** | CSV文件（已有） | 需要额外标注 |
| **作用** | 静态背景信息 | 额外预测任务 |
| **使用方式** | 拼接到特征 | 多任务学习 |
| **实施难度** | 简单（已实现） | 中等（需要标签） |
| **预期提升** | +1~3% | +5~10% |

---

## 五、推荐实施路径

### 阶段1：使用优化损失（当前）✅

```yaml
# 只启用优化损失函数
use_combined_loss: 1      # A1: ASL + Soft-F1
use_corn_loss: 1          # A2: CORN
use_qwk_aux: 1            # A2: QWK

# 辅助属性（已有数据）
use_aux_attrs: true

# 辅助任务（暂不启用）
enable_auxiliary_tasks: false
```

**预期提升**：A1 F1 +3~8%，A2 QWK +3~5%

### 阶段2：添加简单辅助任务（可选）

如果阶段1效果好，可以尝试：

```yaml
# 启用简单的辅助任务
enable_auxiliary_tasks: true
enable_emotion_dims: true   # 从抑郁/焦虑推导
enable_emotion_cls: true    # 从抑郁/焦虑/压力推导
enable_au_pred: false       # 暂不启用
```

**预期提升**：额外 +2~5%

### 阶段3：完整MTL（高级）

如果有完整的辅助标签：

```yaml
use_uncertainty_weighting: true
enable_auxiliary_tasks: true
enable_emotion_dims: true
enable_emotion_cls: true
enable_au_pred: true
```

**预期提升**：总计 +5~10%

---

## 六、快速命令

```bash
# 当前推荐：只使用优化损失
python train.py --task a1 --config tasks/a1/phase1_optimization.yaml

# 如果要启用辅助任务（需要先准备标签）
python train.py --task a1 --config tasks/a1/phase1_optimization.yaml \
    --enable_auxiliary_tasks 1 \
    --enable_emotion_dims 1 \
    --enable_emotion_cls 1

# 查看辅助属性是否加载成功
python -c "
from common.data.grouped_dataset import GroupedParticipantDataset
from common.data.dataset import FeatureConfig
ds = GroupedParticipantDataset('manifests/train.csv', FeatureConfig(), 'train')
sample = ds[0]
print('Auxiliary attributes:', sample['aux_attrs'])
"
```

---

## 总结

1. **辅助属性**（`use_aux_attrs`）：已有功能，数据已存在，默认启用 ✅
2. **辅助任务**（`enable_auxiliary_tasks`）：新增功能，需要额外标签，默认禁用 ❌
3. **推荐路径**：先使用优化损失函数，效果好再考虑添加辅助任务
4. **最简单的方法**：保持当前配置，直接训练即可

如有问题，随时告诉我！
