# grouped_model.py - 参与者聚合模型详解

## 文件概述

`grouped_model.py` 实现了参与者级别的多会话模型，包括：
- 会话特征聚合（将4个会话的特征融合为参与者级别表示）
- 会话类型分类（辅助任务）
- CORAL预测头（可选）

## 设计动机

**为什么需要参与者级别聚合？**

在 AdoDAS 任务中：
- 每个参与者有4个不同类型的会话（A01朗读、B01日常、B02开心、B03悲伤）
- 评估是基于参与者的整体心理健康状态
- 单个会话可能不足以反映整体状态
- 多个会话的信息可以相互补充和验证

**多会话聚合的好处**：
1. **信息互补**：不同会话捕获不同情感状态
2. **鲁棒性**：单个会话异常可以被其他会话弥补
3. **一致性**：模型被鼓励产生跨会话一致的特征

## ParticipantAggregator - 参与者聚合器

### 类结构

```python
class ParticipantAggregator(nn.Module):
    def __init__(
        self,
        d_in: int,          # 输入维度 (d_shared)
        d_out: int,         # 输出维度 (通常等于 d_in)
        method: str = "mlp",  # 聚合方法
        dropout: float = 0.2
    ):
        self.method = method
        # 支持三种聚合方法: "mean", "mlp", "attention"
```

### 聚合方法详解

#### 方法1: Mean (简单平均)

```python
if self.method == "mean":
    n_valid = mask.sum(dim=1).clamp(min=1)  # 有效会话数
    pooled = masked_reprs.sum(dim=1) / n_valid  # 平均
    return self.proj(pooled)
```

**原理**：对所有有效会话的特征取平均

**公式**：
```
pooled = Σ(valid_i × repr_i) / n_valid
```

**优点**：
- 简单高效
- 对所有会话一视同仁
- 无额外参数

**缺点**：
- 无法区分会话重要性
- 对异常会话敏感

**适用场景**：
- 会话质量均匀
- 计算资源有限

#### 方法2: MLP (多层感知机)

```python
elif self.method == "mlp":
    n_valid = mask.sum(dim=1).clamp(min=1)
    pooled = masked_reprs.sum(dim=1) / n_valid  # 先平均
    return self.mlp(pooled)  # 再通过MLP转换
```

**MLP 结构**：
```python
self.mlp = nn.Sequential(
    nn.Linear(d_in, d_out),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(d_out, d_out),
)
```

**原理**：
1. 先对所有会话平均
2. 通过MLP学习非线性变换

**优点**：
- 学习非线性组合
- 参数适中（约 d_in × d_out × 2）
- 表达能力优于简单平均

**缺点**：
- 仍然对所有会话一视同仁
- 无法显式建模会话间关系

**适用场景**：
- 默认推荐
- 平衡效率和性能

#### 方法3: Attention (注意力机制)

```python
elif self.method == "attention":
    scores = self.query(session_reprs).squeeze(-1)  # 计算注意力分数
    scores = scores.masked_fill(~session_valid, float("-inf"))  # 屏蔽无效会话
    weights = F.softmax(scores, dim=-1)  # softmax归一化
    weights = weights.masked_fill(~session_valid, 0.0)  # 再次屏蔽
    pooled = (weights.unsqueeze(-1) * session_reprs).sum(dim=1)  # 加权求和
    return self.proj(pooled)
```

**结构**：
```python
self.query = nn.Linear(d_in, 1)  # 为每个会话计算一个注意力分数
self.proj = nn.Linear(d_in, d_out)  # 输出投影
```

**计算流程**：

```
输入: session_reprs (B, 4, d_in), session_valid (B, 4)

1. 计算注意力分数:
   scores = Linear(session_reprs).squeeze(-1)  → (B, 4)
   scores = [2.5, 1.3, 0.8, -inf]  (假设第4个无效)

2. Softmax归一化:
   weights = softmax(scores) = [0.55, 0.30, 0.15, 0.00]

3. 加权求和:
   pooled = Σ(weights[i] × session_reprs[i])  → (B, d_in)

4. 投影输出:
   output = proj(pooled)  → (B, d_out)
```

**优点**：
- 自动学习会话重要性
- 对有效会话分配不同权重
- 更灵活

**缺点**：
- 额外参数
- 可能过拟合（当会话数少时）

**适用场景**：
- 会话质量差异大
- 数据量充足

### 聚合方法对比

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        聚合方法对比表                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  方法      │  参数量   │  会话权重   │  适用场景                              │
│  ─────────────────────────────────────────────────────────────────────────  │
│  mean      │  d_in×d_out  │  相等      │  会话质量均匀，资源有限                │
│  mlp       │  ~2×d_in×d_out  │  相等   │  默认推荐，平衡效率性能                │
│  attention │  d_in+ d_in×d_out  │  可学习  │  会话质量差异大，数据充足           │
│                                                                             │
│  示例 (d_in=256, d_out=256):                                                 │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  mean:      256 × 256 = 65,536                                        │  │
│  │  mlp:       256×256 + 256×256 = 131,072                               │  │
│  │  attention: 256 + 256×256 = 65,792                                    │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## SessionTypeClassifier - 会话类型分类器

### 设计动机

**为什么需要会话类型分类？**

这是一个**辅助任务（auxiliary task）**，目的：
1. **强迫模型学习会话语义差异**：让模型区分A01(朗读)、B01(日常)、B02(开心)、B03(悲伤)
2. **正则化效果**：防止主任务过拟合
3. **多任务学习**：共享表示提升泛化

### 结构

```python
class SessionTypeClassifier(nn.Module):
    def __init__(self, d_in: int, n_classes: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_in, 64),
            nn.GELU(),
            nn.Linear(64, n_classes),
        )
```

### 输入输出

```
输入: session_repr (B×4, d_in)  - 每个会话的特征表示
输出: logits (B×4, 4)           - 4个会话类型的logit

会话类型标签：
- 0: A01 (朗读)
- 1: B01 (日常)
- 2: B02 (开心)
- 3: B03 (悲伤)
```

### 损失计算

```python
# 只对有效会话计算损失
valid_session_mask = _flatten_valid_session_mask(session_valid)
type_loss = F.cross_entropy(
    out["session_type_logits"][valid_session_mask],
    session_types[valid_session_mask]
)
```

## GroupedModel - 分组参与者模型

### 整体架构

```python
class GroupedModel(nn.Module):
    def __init__(
        self,
        backbone: MTCNBackbone,
        d_shared: int,
        aggregator_method: str = "mlp",
        dropout: float = 0.2,
        aux_encoder=None,          # AuxiliaryAttributeEncoder (输入端)
        aux_heads=None,            # AuxAttributeHeads (LUPI Phase 1 输出端)
    ):
        super().__init__()
        self.backbone = backbone
        self.aggregator = ParticipantAggregator(d_shared, d_shared, aggregator_method, dropout)
        self.session_type_head = SessionTypeClassifier(d_in=d_shared, n_classes=4)
        self.aux_encoder = aux_encoder
        self.aux_heads = aux_heads
```

### 前向传播流程

```python
def forward(
    self,
    flat_batch: dict,                    # 展平后的批次数据
    n_participants: int,                 # 参与者数量 B
    session_valid: torch.Tensor,         # 会话有效性掩码 (B, 4)
) -> dict[str, torch.Tensor]:
```

**步骤详解**：

```
输入:
  flat_batch: 包含所有会话的展平特征 (N=B×4个样本)
  n_participants: B
  session_valid: (B, 4) - 标记哪些会话有效

1. 骨干网络前向:
   session_reprs = self.backbone(flat_batch)  → (N, d_shared)

2. 重塑为参与者网格:
   session_grid = session_reprs.view(B, 4, -1)  → (B, 4, d_shared)

3. 参与者聚合:
   participant_repr = self.aggregator(session_grid, session_valid)  → (B, d_shared)

   3a. LUPI Phase 1: 辅助属性预测 (aux_encoder 拼接之前)
       if self.aux_heads is not None:
           aux_logits = self.aux_heads(participant_repr)  → {name: (B, n_classes)}

   3b. 辅助属性编码拼接 (输入端, 可选)
       if self.aux_encoder is not None:
           participant_repr = cat([participant_repr, aux_encoded])  → (B, d_shared + aux_dim)

4. 会话类型分类:
   session_type_logits = self.session_type_head(session_reprs)  → (N, 4)

5. 返回:
   {
       "session_reprs": session_reprs,       # (N, d_shared)
       "participant_repr": participant_repr, # (B, d_shared [+ aux_dim])
       "session_type_logits": session_type_logits,  # (N, 4)
       "aux_logits": aux_logits,             # LUPI: Dict or None
   }
```

### 数据流可视化

```
输入批次 (B个参与者):
┌─────────────────────────────────────────────────────────────────────────────┐
│  Participant 1              Participant 2               ...                │
│  ┌─────────┬─────────┬─────────┬─────────┐                                   │
│  │ A01     │ B01     │ B02     │ B03     │                                   │
│  │ (valid) │ (valid) │ (invalid)│ (valid)│                                   │
│  └─────────┴─────────┴─────────┴─────────┘                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                          ↓
                          │
                          ▼
展平批次 (flat_batch):
┌─────────────────────────────────────────────────────────────────────────────┐
│  [A01_1, B01_1, B02_1, B03_1, A01_2, B01_2, ...]  (N = B×4 个样本)          │
└─────────────────────────────────────────────────────────────────────────────┘
                          ↓
                          │  MTCNBackbone 处理
                          ▼
会话特征 (session_reprs):
┌─────────────────────────────────────────────────────────────────────────────┐
│  (N, d_shared)                                                              │
│  ┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐             │
│  │ repr_A01│ repr_B01│ repr_B02│ repr_B03│ repr_A01│ repr_B01│             │
│  │    1    │    1    │    1    │    1    │    2    │    2    │             │
│  └─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘             │
└─────────────────────────────────────────────────────────────────────────────┘
                          ↓
                          │  reshape(B, 4, -1)
                          ▼
会话网格 (session_grid):
┌─────────────────────────────────────────────────────────────────────────────┐
│  (B, 4, d_shared)                                                           │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  [[repr_A01_1, repr_B01_1, repr_B02_1, repr_B03_1], ...]            │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                          ↓
                          │  ParticipantAggregator
                          ▼
参与者表示 (participant_repr):
┌─────────────────────────────────────────────────────────────────────────────┐
│  (B, d_shared)                                                              │
│  ┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐             │
│  │ repr_p1 │ repr_p2 │ repr_p3 │ ...                                        │
│  └─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 多任务损失

```python
# 主任务损失（参与者级别）
main_loss = task_loss(participant_repr, participant_targets)

# 会话级辅助损失
s_logits = task_head(session_reprs)[valid_session_mask]
s_targets = targets.unsqueeze(1).expand(-1, 4, -1).reshape(-1, ...)[valid_session_mask]
sess_loss = task_loss(s_logits, s_targets)

# 会话类型分类损失
type_loss = F.cross_entropy(
    session_type_logits[valid_session_mask],
    session_types[valid_session_mask]
)

# 总损失
loss = main_loss + w1 × sess_loss + w2 × type_loss
```

**默认权重**：
- session_loss_weight = 0.5
- session_type_loss_weight = 0.15

## 训练流程中的使用

### 训练阶段

```python
# 1. 前向传播
out = grouped_model(flat_batch, B, session_valid)

# 2. 计算主任务损失
p_logits = task_head(out["participant_repr"])
main_loss = a1_loss(p_logits, targets)  # 或 a2_ordinal_loss

# 3. 计算会话级辅助损失（可选）
s_logits = task_head(out["session_reprs"])[valid_session_mask]
s_targets = targets.unsqueeze(1).expand(-1, 4, -1).reshape(-1, 3)[valid_session_mask]
sess_loss = a1_loss(s_logits, s_targets)

# 4. 计算会话类型损失
type_loss = F.cross_entropy(
    out["session_type_logits"][valid_session_mask],
    session_types[valid_session_mask]
)

# 5. 总损失
loss = main_loss + 0.5 * sess_loss + 0.15 * type_loss
```

### 推理阶段

```python
# 参与者级别预测
out = grouped_model(flat_batch, B, session_valid)
participant_logits = task_head(out["participant_repr"])

# 或使用会话级别预测（可选）
session_logits = task_head(out["session_reprs"])
```

## 设计考量

### 为什么展平处理？

**优势**：
1. **并行处理**：所有会话可以并行通过骨干网络
2. **简化实现**：无需处理变长序列和可变会话数
3. **GPU友好**：连续内存访问更高效

**实现细节**：
```python
# 在 grouped_collate_fn 中展平
for b_idx, sample in enumerate(batch):
    for s_idx, sess_data in enumerate(sample["sessions"]):
        if sess_data is not None:
            all_sessions.append(sess_data)
            session_types.append(s_idx)
```

### 如何处理缺失会话？

**策略**：
1. 使用虚拟会话（dummy session）填充
2. 通过 `session_valid` 掩码标记
3. 在聚合时忽略无效会话

```python
# 创建虚拟会话
def _make_dummy_session(ref: dict) -> dict:
    T = 1  # 最小长度
    return {
        "audio_groups": {k: torch.zeros(T, v.shape[-1]) for k, v in ref["audio_groups"].items()},
        "video_groups": {k: torch.zeros(T, v.shape[-1]) for k, v in ref["video_groups"].items()},
        "mask_audio": torch.zeros(T, dtype=torch.bool),
        "mask_video": torch.zeros(T, dtype=torch.bool),
        # ...
    }

# 聚合时过滤
masked_reprs = session_reprs * mask.float()
pooled = masked_reprs.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
```

## 完整使用示例

```python
from common.models.mtcn_backbone import MTCNBackbone, BackboneConfig
from common.models.grouped_model import GroupedModel
from common.models.heads import A1Head

# 1. 配置骨干网络
dims = {"mel_mfcc": 60, "vad": 1, "ssl_embed": 768, ...}
bb_cfg = BackboneConfig(
    audio_group_dims={n: dims[n] for n in ["mel_mfcc", "vad", "ssl_embed"]},
    audio_pooled_group_dims={"egemaps": 88},
    video_group_dims={n: dims[n] for n in ["headpose_geom", ...]},
    d_adapter=64,
    d_model=256,
    d_shared=256,
    tcn_layers=6,
    dropout=0.2,
)

# 2. 创建模型
backbone = MTCNBackbone(bb_cfg)
grouped_model = GroupedModel(
    backbone=backbone,
    d_shared=256,
    aggregator_method="mlp",  # 或 "mean", "attention"
    dropout=0.2,
)

# 3. 创建任务头
task_head = A1Head(d_in=256)

# 4. 前向传播
flat_batch = collate_fn(batch)["flat_batch"]
session_valid = collate_fn(batch)["session_valid"]
B = collate_fn(batch)["n_participants"]

out = grouped_model(flat_batch, B, session_valid)
logits = task_head(out["participant_repr"])  # (B, 3)
```