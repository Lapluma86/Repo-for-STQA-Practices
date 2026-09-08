# heads.py - 任务预测头详解

## 文件概述

`heads.py` 定义了任务预测头和辅助属性预测头：A1Head、A2OrdinalHead、CORALHead，以及 LUPI Phase 1 的 AuxAttributeHeads。此外还包含损失函数：ASL、Soft-F1、CORN、QWK 和 aux_attribute_loss。

## A1Head - 三分类二元预测头

### 模型结构

```python
class A1Head(nn.Module):
    def __init__(self, d_in: int, bias_init: list[float] | None = None):
        super().__init__()
        self.fc = nn.Linear(d_in, 3)  # 输出3个logits (D, A, S)
        
        # 可选：根据训练集正样本率初始化偏置
        if bias_init is not None:
            with torch.no_grad():
                self.fc.bias.copy_(torch.tensor(bias_init, dtype=torch.float32))
```

### 偏置初始化原理

**为什么需要特殊的偏置初始化？**

在心理健康数据中，正样本（抑郁、焦虑、压力）通常比例较低（如10-20%）。如果使用默认偏置（0），模型初期会预测所有样本为负类，导致训练不稳定。

**解决方案：使用先验概率初始化偏置**

```python
def _compute_bias_init_a1(manifest_path: Path) -> list[float]:
    """根据训练集正样本率计算初始偏置"""
    df = pd.read_csv(manifest_path)
    biases = []
    for col in ["y_D", "y_A", "y_S"]:
        # 计算正样本率
        rate = df[col].mean()
        # 限制在合理范围，避免极端值
        rate = max(min(rate, 0.99), 0.01)
        # 计算对数偏置: log(p / (1-p))
        biases.append(math.log(rate / (1 - rate)))
    return biases
```

**数学推导**：

对于二元分类，sigmoid 函数为：
```
p = sigmoid(logit) = 1 / (1 + exp(-logit))
```

反解得到：
```
logit = log(p / (1 - p))
```

如果我们想让模型初始预测概率等于训练集正样本率 `rate`：
```
初始偏置 = log(rate / (1 - rate))
```

示例：如果 rate = 0.15（15%正样本）
```
bias_init = log(0.15 / 0.85) = log(0.176) ≈ -1.74
```

这意味着初始 logit = -1.74，sigmoid(-1.74) ≈ 0.15，恰好等于正样本率！

### 损失函数

```python
def a1_loss(
    logits: torch.Tensor,        # (B, 3)
    targets: torch.Tensor,       # (B, 3) 0或1
    pos_weight: torch.Tensor | None = None,  # 正样本权重
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    if label_smoothing > 0.0:
        # 标签平滑: 将硬标签软化
        targets = targets.float() * (1.0 - label_smoothing) + 0.5 * label_smoothing
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
```

### 标签平滑原理

**目的**：防止模型对训练数据过度自信

**实现**：
```python
原始标签: [0, 1]  →  平滑标签: [label_smoothing/2, 1 - label_smoothing/2]
```

示例（label_smoothing = 0.05）：
```
原始标签 y=0 → 平滑后 y=0.025
原始标签 y=1 → 平滑后 y=0.975
```

### 正样本权重

**解决类别不平衡**：

```python
def _compute_pos_weight_a1(manifest_path: Path) -> list[float]:
    df = pd.read_csv(manifest_path)
    weights = []
    for col in ["y_D", "y_A", "y_S"]:
        n_pos = df[col].sum()      # 正样本数
        n_neg = len(df) - n_pos    # 负样本数
        # 计算权重，使用平方根降低极端值影响
        w = float(np.sqrt(n_neg / max(n_pos, 1)))
        # 限制范围 [1.0, 4.0]
        w = max(1.0, min(w, 4.0))
        weights.append(w)
    return weights
```

**损失计算**：
```python
loss = BCEWithLogits(logits, targets)
     = -[w_pos × y × log(sigmoid(logit)) + w_neg × (1-y) × log(1-sigmoid(logit))]
```

其中 `w_pos = pos_weight`，`w_neg = 1.0`

### 预测输出

```python
@staticmethod
def predict_probs(logits: torch.Tensor) -> torch.Tensor:
    """将logits转换为概率"""
    return torch.sigmoid(logits)

# 输出: (B, 3) 的概率值，每个值在 [0, 1] 范围
# 分别表示 p_D (抑郁概率), p_A (焦虑概率), p_S (压力概率)
```

## A2OrdinalHead - 序数回归预测头

### 序数回归原理

**问题背景**：
- A2任务要求预测21个评估项目，每个取值 {0, 1, 2, 3}
- 这些分数是有序的：0 < 1 < 2 < 3
- 直接用4类分类会丢失序数信息

**解决方案：阈值分解**

将序数问题分解为多个阈值二元分类：
```
分数 >= 1 ?  → 第1个阈值
分数 >= 2 ?  → 第2个阈值
分数 >= 3 ?  → 第3个阈值
```

如果分数 = 2：
```
>= 1: True (阈值1通过)
>= 2: True (阈值2通过)
>= 3: False (阈值3未通过)
```

通过计数可得分数 = 2。

### 模型结构

```python
class A2OrdinalHead(nn.Module):
    def __init__(self, d_in: int, n_items: int = 21, n_thresholds: int = 3):
        super().__init__()
        self.n_items = n_items        # 21个评估项目
        self.n_thresholds = n_thresholds  # 3个阈值
        self.fc = nn.Linear(d_in, n_items * n_thresholds)  # 输出63个值
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        # 重塑为 (B, 21, 3)
        return self.fc(x).view(B, self.n_items, self.n_thresholds)
```

**输出解释**：
```
logits[b, i, k] = 第b个样本，第i个项目，第k个阈值的logit值
其中 k=0 表示 >=1, k=1 表示 >=2, k=2 表示 >=3
```

### 目标构建

```python
@staticmethod
def build_ordinal_targets(labels: torch.Tensor, n_thresholds: int = 3) -> torch.Tensor:
    """
    将整数标签转换为序数目标
    
    输入: labels (B, 21), 整数 0-3
    输出: targets (B, 21, 3), 每个阈值是否通过
    """
    B, I = labels.shape
    thresholds = torch.arange(1, n_thresholds + 1, device=labels.device).float()
    # thresholds = [1, 2, 3]
    
    # 对于每个标签，检查是否 >= 每个阈值
    targets = (labels.unsqueeze(-1).float() >= thresholds.view(1, 1, -1)).float()
    
    return targets
```

示例：
```
标签 = 2
thresholds = [1, 2, 3]

targets = (2 >= 1, 2 >= 2, 2 >= 3) = (True, True, False) = [1, 1, 0]
```

### 损失函数

```python
def a2_ordinal_loss(
    logits: torch.Tensor,        # (B, 21, 3)
    labels: torch.Tensor,        # (B, 21) 整数0-3
    pos_weight: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    # 构建序数目标
    targets = A2OrdinalHead.build_ordinal_targets(labels, n_thresholds=logits.size(-1))
    
    if label_smoothing > 0.0:
        targets = targets * (1.0 - label_smoothing) + 0.5 * label_smoothing
    
    # 二元交叉熵损失
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
```

### 解码方法详解

**方法1: argmax (简单计数)**
```python
@staticmethod
def predict_int(logits: torch.Tensor) -> torch.Tensor:
    """
    简单计数方法
    
    计算每个阈值 sigmoid > 0.5 的数量
    """
    # sigmoid(logits) 得到每个阈值通过的"概率"
    # > 0.5 判断是否通过
    # .sum(dim=-1) 计算通过的阈值数
    return (torch.sigmoid(logits) > 0.5).long().sum(dim=-1)
```

示例：
```
logits = [2.0, 1.5, -0.5]  # 三个阈值的logit
sigmoid(logits) = [0.88, 0.82, 0.38]
> 0.5 = [True, True, False]
sum = 2  → 预测分数为2
```

**方法2: monotonic (强制单调)**
```python
@staticmethod
def predict_int_monotonic(logits: torch.Tensor) -> torch.Tensor:
    """
    强制单调递减的概率
    
    原理: 如果分数 >= k，必然也 >= (k-1)
    所以 P(>=k) <= P(>=k-1)
    """
    s = torch.sigmoid(logits)
    
    # 强制单调: p_k <= p_{k-1}
    p1 = s[..., 0]
    p2 = torch.min(s[..., 1], p1)  # p2 <= p1
    p3 = torch.min(s[..., 2], p2)  # p3 <= p2
    
    # 计算各分数的概率
    P0 = 1.0 - p1        # P(score=0) = P(<1) = 1 - P(>=1)
    P1 = p1 - p2         # P(score=1) = P(>=1) - P(>=2)
    P2 = p2 - p3         # P(score=2) = P(>=2) - P(>=3)
    P3 = p3              # P(score=3) = P(>=3)
    
    class_probs = torch.stack([P0, P1, P2, P3], dim=-1)
    return class_probs.argmax(dim=-1)  # 选择概率最大的分数
```

**方法3: expectation (期望值)**
```python
@staticmethod
def predict_expectation(logits: torch.Tensor) -> torch.Tensor:
    """
    使用期望值作为预测
    
    原理: 分数的期望值 = P(>=1) + P(>=2) + P(>=3)
    """
    s = torch.sigmoid(logits)
    
    # 强制单调
    p1 = s[..., 0]
    p2 = torch.min(s[..., 1], p1)
    p3 = torch.min(s[..., 2], p2)
    
    # 期望值
    E = p1 + p2 + p3
    
    # 四舍五入并限制在 [0, 3]
    return E.round().long().clamp(0, 3)
```

**三种方法对比**：

```
logits = [2.0, 1.5, -0.5]

方法1 argmax:
  sigmoid = [0.88, 0.82, 0.38]
  > 0.5 = [1, 1, 0]
  sum = 2  → 预测 = 2

方法2 monotonic:
  sigmoid = [0.88, 0.82, 0.38]
  monotonic: p1=0.88, p2=min(0.82,0.88)=0.82, p3=min(0.38,0.82)=0.38
  P0=1-0.88=0.12, P1=0.88-0.82=0.06, P2=0.82-0.38=0.44, P3=0.38
  argmax = 2  → 预测 = 2

方法3 expectation:
  monotonic probs: p1=0.88, p2=0.82, p3=0.38
  E = 0.88 + 0.82 + 0.38 = 2.08
  round(2.08) = 2  → 预测 = 2
```

## CORALHead - Consistent Rank Logits

### CORAL 原理

**问题**：A2OrdinalHead 的阈值是隐式学习的，可能不保证单调性

**CORAL 解决方案**：
- 显式学习一个分数 `score` 和可学习的阈值 `thresholds`
- logit = score - threshold
- 阈值通过 cumsum(softplus) 保证单调递增

### 模型结构

```python
class CORALHead(nn.Module):
    def __init__(self, d_in: int, n_items: int = 21, n_thresholds: int = 3):
        super().__init__()
        self.n_items = n_items
        self.n_thresholds = n_thresholds
        
        # 分数预测层
        self.score_fc = nn.Linear(d_in, n_items)  # 输出21个分数
        
        # 可学习的阈值间距
        self.raw_thresholds = nn.Parameter(torch.zeros(n_items, n_thresholds))
        nn.init.constant_(self.raw_thresholds, 0.5)  # 初始间距
```

### 前向传播

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    # 1. 预测分数
    scores = self.score_fc(x)  # (B, 21)
    
    # 2. 计算阈值（保证单调递增）
    spacings = F.softplus(self.raw_thresholds)  # (21, 3) 间距 > 0
    thresholds = torch.cumsum(spacings, dim=-1)  # (21, 3) 累加保证单调
    
    # 3. 计算 logits
    logits = scores.unsqueeze(-1) - thresholds.unsqueeze(0)
    # (B, 21, 1) - (1, 21, 3) = (B, 21, 3)
    
    return logits
```

**阈值计算详解**：

```
raw_thresholds = [0.5, 0.5, 0.5]  (初始间距)

softplus(0.5) ≈ 0.97  (保证正值)

spacings = [0.97, 0.97, 0.97]

cumsum → thresholds = [0.97, 1.94, 2.91]  (单调递增!)

logits = score - thresholds
       = score - [0.97, 1.94, 2.91]
       = [score - 0.97, score - 1.94, score - 2.91]
```

**如果 score = 2.0**:
```
logits = [2.0 - 0.97, 2.0 - 1.94, 2.0 - 2.91]
       = [1.03, 0.06, -0.91]
sigmoid → [0.74, 0.52, 0.29]

分数 >= 1? sigmoid(1.03) = 0.74 > 0.5 → Yes
分数 >= 2? sigmoid(0.06) = 0.52 > 0.5 → Yes (勉强)
分数 >= 3? sigmoid(-0.91) = 0.29 < 0.5 → No

预测分数 = 2
```

### CORAL vs OrdinalHead 对比

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     CORALHead vs A2OrdinalHead                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  A2OrdinalHead:                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  - 直接预测所有 logits                                                │  │
│  │  - 阈值隐式包含在 logit 中                                            │  │
│  │  - 可能出现非单调情况 (sigmoid不保证递减)                              │  │
│  │  - 参数量: d_in × 21 × 3                                              │  │
│  │  - 需要后处理强制单调                                                  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  CORALHead:                                                                 │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  - 分离分数和阈值                                                     │  │
│  │  - 阈值通过 cumsum(softplus) 保证单调                                  │  │
│  │  - 参数量: d_in × 21 + 21 × 3                                         │  │
│  │  - 数学上更优雅，更符合序数假设                                        │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  参数量对比 (d_in=256):                                                      │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  A2OrdinalHead: 256 × 63 = 16,128                                     │  │
│  │  CORALHead: 256 × 21 + 63 = 5,376 + 63 = 5,439                        │  │
│  │  CORALHead 参数量更少！                                                │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## AuxAttributeHeads - LUPI 辅助属性预测头

### 设计动机

LUPI (Learning Using Privileged Information) Phase 1：训练时从 participant_repr (纯音视频表示) 预测 5 个辅助属性，迫使 backbone 编码与心理状态相关的潜变量。推理时不依赖辅助属性。

与 `AuxiliaryAttributeEncoder` (输入端编码) 互补——Encoder 将辅助属性编码为输入特征，AuxAttributeHeads 反向预测辅助属性作为多任务监督信号。

### 模型结构

```python
_AUX_ATTR_SPEC = {
    "aux_family":     {"num_classes": 6, "label_offset": -1},  # raw 1-6 → label 0-5
    "aux_only_child": {"num_classes": 2, "label_offset":  0},  # raw 0-1 → label 0-1
    "aux_favoritism": {"num_classes": 3, "label_offset": -1},  # raw 1-3 → label 0-2
    "aux_academic":   {"num_classes": 3, "label_offset": -1},  # raw 1-3 → label 0-2
    "aux_emotional":  {"num_classes": 3, "label_offset": -1},  # raw 1-3 → label 0-2
}

class AuxAttributeHeads(nn.Module):
    def __init__(self, d_in: int, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(d_in, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, spec["num_classes"]),
            )
            for name, spec in _AUX_ATTR_SPEC.items()
        })

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {name: head(x) for name, head in self.heads.items()}
```

### 损失函数

```python
def aux_attribute_loss(
    aux_logits: dict[str, torch.Tensor],  # {name: (B, n_classes)}
    aux_attrs: torch.Tensor,              # (B, 5) 原始值, -1=缺失
    weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """加权 CrossEntropy, 自动 skip -1 缺失值"""
    total = 0.0
    acc_dict = {}
    for i, name in enumerate(_AUX_ATTR_NAMES):
        valid = aux_attrs[:, i] >= 0     # 过滤缺失值
        if valid.sum() == 0: continue
        labels = (aux_attrs[:, i] + offset).clamp(min=0)  # 重新映射到 0-indexed
        ce = F.cross_entropy(aux_logits[name][valid], labels[valid])
        total += weights[name] * ce
        acc_dict[name] = accuracy(logits[valid], labels[valid])
    return total, acc_dict
```

### 数据流位置

```
participant_repr (B, d_shared)
    ├── → AuxAttributeHeads → aux_logits → aux_attribute_loss (训练时监督信号)
    └── → concat(aux_encoded) → 任务头 → 主任务预测
```

关键设计：AuxAttributeHeads 作用于 **aux_encoder 拼接之前**的 participant_repr（纯音视频表示），确保 LUPI 范式正确——backbone 在无辅助属性时也能学到相关表示。

### 权重推荐

基于临床先验：
- `aux_emotional`: 0.20 (与 DASS 最相关)
- `aux_academic`: 0.15 (抑郁前驱症状)
- `aux_family` / `aux_only_child` / `aux_favoritism`: 0.05 (弱相关)

总辅助损失应为主任务的 1/3 ~ 1/2。若主指标下降，整体下调权重 50%。

## 使用示例

### A1Head 使用

```python
from common.models.heads import A1Head, a1_loss

# 创建预测头（带偏置初始化）
bias_init = [-1.74, -1.50, -1.60]  # 根据训练集计算
head = A1Head(d_in=256, bias_init=bias_init)

# 前向传播
participant_repr = model.backbone_output  # (B, 256)
logits = head(participant_repr)  # (B, 3)

# 计算损失
targets = batch["y_a1"]  # (B, 3)
pos_weight = torch.tensor([3.5, 2.8, 3.2])  # 根据不平衡率计算
loss = a1_loss(logits, targets, pos_weight=pos_weight, label_smoothing=0.05)

# 预测概率
probs = A1Head.predict_probs(logits)
predictions = (probs > 0.5).long()  # 二元预测
```

### A2OrdinalHead 使用

```python
from common.models.heads import A2OrdinalHead, a2_ordinal_loss

# 创建预测头
head = A2OrdinalHead(d_in=256)

# 前向传播
logits = head(participant_repr)  # (B, 21, 3)

# 计算损失
labels = batch["y_a2"]  # (B, 21) 整数0-3
loss = a2_ordinal_loss(logits, labels, label_smoothing=0.05)

# 解码预测（三种方法）
pred_argmax = A2OrdinalHead.predict_int(logits)
pred_monotonic = A2OrdinalHead.predict_int_monotonic(logits)
pred_expectation = A2OrdinalHead.predict_expectation(logits)
```

### CORALHead 使用

```python
from common.models.grouped_model import CORALHead

# 创建预测头
head = CORALHead(d_in=256)

# 前向传播
logits = head(participant_repr)  # (B, 21, 3)

# 解码预测
pred_int = CORALHead.predict_int(logits)
pred_monotonic = CORALHead.predict_int_monotonic(logits)
pred_expectation = CORALHead.predict_expectation(logits)
```