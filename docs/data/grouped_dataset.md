# grouped_dataset.py - 多会话分组数据集详解

## 文件概述

`grouped_dataset.py` 扩展了单会话数据处理，实现了**参与者级**（Participant-level）的数据组织。每个参与者包含4个会话（A01, B01, B02, B03），这种设计支持从多个会话中聚合特征进行预测。

## 设计原理

### 为什么需要多会话数据？

1. **多模态综合评估**：
   - A01（朗读）：语音韵律特征
   - B01（日常）：叙事能力
   - B02（开心）：正面情绪表达
   - B03（悲伤）：负面情绪表达
   
2. **增强泛化性**：
   - 单一会话可能受特定情境影响
   - 多个会话提供更全面的心理健康画像

3. **辅助任务学习**：
   - 会话类型分类作为辅助任务
   - 参与者级 + 会话级联合训练

## GroupedParticipantDataset 类

### 参与者数据结构

```python
class GroupedParticipantDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        cfg: FeatureConfig,
        split: str,
        session_drop_prob: float = 0.0,  # 训练时随机丢弃会话的概率
    ):
```

**数据分组逻辑**：
```python
# 读取manifest并按(学校, 班级, 参与者ID)分组
manifest = pd.read_csv(manifest_path)
group_cols = ["anon_school", "anon_class", "anon_pid"]
grouped = manifest.groupby(group_cols)

# 为每个参与者收集会话信息
self.participants = []
for (school, cls, pid), group in grouped:
    sess_rows = {}
    for _, row in group.iterrows():
        sess = str(row["session"])
        sess_rows[sess] = row  # 保存该行数据
    
    # 提取标签（从任意一行，因为所有会话标签相同）
    any_row = group.iloc[0]
    y_a1 = np.array([float(any_row.get(c, -1)) for c in A1_COLS])
    y_a2 = np.array([float(any_row.get(c, -1)) for c in ITEM_COLS])
    
    self.participants.append({
        "anon_school": str(school),
        "anon_class": str(cls),
        "anon_pid": str(pid),
        "sess_rows": sess_rows,  # {session_name: row_data}
        "y_a1": y_a1,
        "y_a2": y_a2,
    })
```

**参与者数据示例**：
```python
{
    "anon_school": "s001",
    "anon_class": "c001",
    "anon_pid": "p001",
    "sess_rows": {
        "A01": Row(...),  # 包含该会话的manifest行
        "B01": Row(...),
        "B02": Row(...),
    },  # 注意: B03可能缺失
    "y_a1": [1.0, 0.0, 1.0],  # 抑郁、焦虑、压力
    "y_a2": [2, 1, 3, ...],   # 21个项目分数
}
```

### 单会话加载

```python
def _load_single_session(self, row) -> dict[str, Any] | None:
    """加载单个会话的特征，失败时返回None"""
```

**加载流程**：
1. 加载音频和视频原始特征组
2. 对齐到统一时间网格
3. 计算模态掩码
4. 提取VAD和QC信号
5. 加载池化特征
6. 填充缺失特征
7. 返回会话数据字典

### 参与者级加载

```python
def _load_participant(self, idx: int) -> dict[str, Any]:
    """加载一个参与者的所有会话"""
    info = self.participants[idx]
    
    sessions_data = []
    session_valid = []
    
    # 按固定顺序加载4个会话
    for sess_name in SESSIONS:  # ["A01", "B01", "B02", "B03"]
        if sess_name in info["sess_rows"]:
            data = self._load_single_session(info["sess_rows"][sess_name])
            if data is not None:
                sessions_data.append(data)
                session_valid.append(True)
            else:
                sessions_data.append(None)
                session_valid.append(False)
        else:
            sessions_data.append(None)
            session_valid.append(False)
    
    return {
        "sessions": sessions_data,        # [session_data, None, session_data, ...]
        "session_valid": np.array(session_valid, dtype=bool),  # [True, False, True, ...]
        "y_a1": torch.from_numpy(info["y_a1"]),
        "y_a2": torch.from_numpy(info["y_a2"]),
        "anon_pid": info["anon_pid"],
        "anon_school": info["anon_school"],
        "anon_class": info["anon_class"],
        "session_names": SESSIONS,
    }
```

### 会话丢弃增强

```python
def _apply_session_dropout(self, sample: dict[str, Any]) -> dict[str, Any]:
    """训练时随机丢弃部分会话，增强泛化能力"""
    
    # 找到所有有效会话的索引
    valid_indices = [
        idx for idx, is_valid in enumerate(sample["session_valid"].tolist())
        if is_valid and sample["sessions"][idx] is not None
    ]
    
    # 条件检查
    if len(valid_indices) <= 1:  # 只有一个或没有有效会话，不丢弃
        return sample
    
    if np.random.random() >= self.session_drop_prob:  # 未达到丢弃概率
        return sample
    
    # 随机选择一个会话丢弃
    drop_idx = int(np.random.choice(valid_indices))
    sessions = list(sample["sessions"])
    sessions[drop_idx] = None
    
    session_valid = np.array(sample["session_valid"], copy=True)
    session_valid[drop_idx] = False
    
    return {**sample, "sessions": sessions, "session_valid": session_valid}
```

**作用**：
- 模拟不完整数据场景（参与者只做了部分会话）
- 强迫模型从部分信息中推断
- 增强对缺失数据的鲁棒性
- 类似 Dropout 的正则化效果

## grouped_collate_fn - 分组批次整理

这是整个数据流程中最复杂的函数，它将多个参与者的会话展平成一个批次。

```python
def grouped_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    输入: B个参与者的数据
    输出: 展平后的批次，所有会话被合并
    """
```

### 展平逻辑详解

```
输入批次:
┌─────────────────────────────────────────────────────────────────────────┐
│  Participant 0:                                                         │
│    sessions: [sess_A01, sess_B01, None, None]                          │
│    session_valid: [True, True, False, False]                           │
│    y_a1: [1.0, 0.0, 1.0]                                                │
│                                                                         │
│  Participant 1:                                                         │
│    sessions: [sess_A01, None, sess_B02, None]                          │
│    session_valid: [True, False, True, False]                           │
│    y_a1: [0.0, 1.0, 0.0]                                                │
│                                                                         │
│  Participant 2:                                                         │
│    sessions: [sess_A01, sess_B01, sess_B02, sess_B03]                  │
│    session_valid: [True, True, True, True]                             │
│    y_a1: [1.0, 1.0, 0.0]                                                │
└─────────────────────────────────────────────────────────────────────────┘
                          ↓
展平后:
┌─────────────────────────────────────────────────────────────────────────┐
│  flat_batch (会话展平):                                                 │
│    有效会话: [p0_A01, p0_B01, p1_A01, p1_B02, p2_A01, p2_B01, ...]     │
│    共 7 个会话 (p0:2 + p1:2 + p2:4)                                    │
│                                                                         │
│  session_valid:                                                         │
│    [[True, True, False, False],                                        │
│     [True, False, True, False],                                        │
│     [True, True, True, True]]                                          │
│                                                                         │
│  participant_y_a1:                                                      │
│    [[1.0, 0.0, 1.0],                                                   │
│     [0.0, 1.0, 0.0],                                                   │
│     [1.0, 1.0, 0.0]]                                                   │
│                                                                         │
│  n_participants: 3                                                      │
└─────────────────────────────────────────────────────────────────────────┘
```

### 代码详解

**1. 收集会话信息**：
```python
all_sessions = []        # 所有有效会话数据
session_types = []       # 会话类型索引（0=A01, 1=B01, 2=B02, 3=B03）
session_valid_list = []  # 每个参与者的会话有效性
flat_pids = []           # 展平的参与者ID
flat_sess_names = []     # 展平的会话名称

for b_idx, sample in enumerate(batch):
    session_valid_list.append(sample["session_valid"])
    
    for s_idx, sess_data in enumerate(sample["sessions"]):
        if sess_data is not None:
            # 有效会话
            all_sessions.append(sess_data)
            session_types.append(s_idx)
            flat_pids.append(sample["anon_pid"])
            flat_sess_names.append(SESSIONS[s_idx])
        else:
            # 缺失会话，创建虚拟数据
            dummy = _make_dummy_session(ref_session)
            all_sessions.append(dummy)
            session_types.append(s_idx)
            flat_pids.append(sample["anon_pid"])
            flat_sess_names.append(SESSIONS[s_idx])
```

**2. 填充和堆叠**：
```python
n_flat = len(all_sessions)  # 展平后的总样本数
T_max = max(s["seq_len"] for s in all_sessions)

def _pad_groups(names, key):
    """填充特征组到最大长度"""
    result = {}
    for n in names:
        D = all_sessions[0][key][n].shape[-1]
        t = torch.zeros(n_flat, T_max, D)
        for i, s in enumerate(all_sessions):
            L = s["seq_len"]
            t[i, :L] = s[key][n]
        result[n] = t
    return result

# 填充掩码
pad_mask = torch.ones(n_flat, T_max, dtype=torch.bool)
for i, s in enumerate(all_sessions):
    pad_mask[i, :s["seq_len"]] = False
```

**3. 构建 flat_batch**：
```python
flat_batch = {
    "audio_groups": _pad_groups(audio_names, "audio_groups"),
    "audio_pooled_groups": {
        name: torch.stack([s["audio_pooled_groups"][name] for s in all_sessions])
        for name in pooled_audio_names
    },
    "video_groups": _pad_groups(video_names, "video_groups"),
    "mask_audio": _pad_1d("mask_audio", torch.bool),
    "mask_video": _pad_1d("mask_video", torch.bool),
    "pad_mask": pad_mask,
    "vad_signal": _pad_1d("vad_signal"),
    "qc_quality": _pad_1d("qc_quality"),
    "session_idx": torch.tensor([s["session_idx"] for s in all_sessions]),
    "seq_len": torch.tensor([s["seq_len"] for s in all_sessions]),
    "anon_pid": flat_pids,
    "session": flat_sess_names,
}
```

**4. 返回完整批次**：
```python
return {
    "flat_batch": flat_batch,                # 展平后的会话批次
    "participant_y_a1": torch.stack([b["y_a1"] for b in batch]),  # (B, 3)
    "participant_y_a2": torch.stack([b["y_a2"] for b in batch]),  # (B, 21)
    "session_valid": torch.from_numpy(np.stack(session_valid_list)),  # (B, 4)
    "session_types": torch.tensor(session_types, dtype=torch.long),   # (n_flat,)
    "n_participants": B,                     # 参与者数量
    "anon_pids": [b["anon_pid"] for b in batch],
    "anon_schools": [b["anon_school"] for b in batch],
    "anon_classes": [b["anon_class"] for b in batch],
    "flat_sessions": flat_sess_names,
    "flat_pids": flat_pids,
}
```

### 虚拟会话生成

```python
def _make_dummy_session(ref: dict[str, Any]) -> dict[str, Any]:
    """创建零填充的虚拟会话，保持维度一致"""
    T = 1  # 最小长度
    return {
        "audio_groups": {
            k: torch.zeros(T, v.shape[-1]) 
            for k, v in ref["audio_groups"].items()
        },
        "audio_pooled_groups": {
            k: torch.zeros_like(v) 
            for k, v in ref["audio_pooled_groups"].items()
        },
        "video_groups": {
            k: torch.zeros(T, v.shape[-1]) 
            for k, v in ref["video_groups"].items()
        },
        "mask_audio": torch.zeros(T, dtype=torch.bool),
        "mask_video": torch.zeros(T, dtype=torch.bool),
        "vad_signal": torch.zeros(T),
        "qc_quality": torch.zeros(T),
        "session_idx": 0,
        "seq_len": T,
        "session": "A01",
    }
```

**为什么需要虚拟会话？**
- 保证批次中所有参与者有相同结构
- 允许模型处理缺失数据
- 通过 `session_valid` 掩码可以忽略虚拟会话
- 避免动态形状导致的复杂处理

## 数据流完整流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Grouped Dataset 数据流                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. Manifest CSV                                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  每个参与者4行 (A01, B01, B02, B03)                                  │   │
│  │  包含相同的 y_a1, y_a2 标签                                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                          ↓                                                  │
│                                                                             │
│  2. 按参与者分组                                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  GroupedParticipantDataset 按 (school, class, pid) 分组              │   │
│  │  每个参与者: {sess_rows, y_a1, y_a2}                                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                          ↓                                                  │
│                                                                             │
│  3. 加载单个参与者                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  _load_participant():                                                │   │
│  │    for sess in [A01, B01, B02, B03]:                                 │   │
│  │        data = _load_single_session(row)  # 使用 dataset.py 的逻辑   │   │
│  │    return {sessions, session_valid, y_a1, y_a2}                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                          ↓                                                  │
│                                                                             │
│  4. 可选：会话丢弃                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  _apply_session_dropout():                                           │   │
│  │    if training and random < session_drop_prob:                       │   │
│  │        随机丢弃一个会话                                              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                          ↓                                                  │
│                                                                             │
│  5. 批次整理 (grouped_collate_fn)                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │    展平所有有效会话                                                  │   │
│  │    填充到最大长度                                                    │   │
│  │    创建虚拟会话（用于缺失位置）                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                          ↓                                                  │
│                                                                             │
│  6. 输入模型                                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  GroupedModel.forward():                                             │   │
│  │    session_reprs = Backbone(flat_batch)  # (B×4, d_shared)          │   │
│  │    session_grid = reshape(B, 4, d_shared)                          │   │
│  │    participant_repr = Aggregator(session_grid, session_valid)      │   │
│  │    return participant_repr, session_reprs                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                          ↓                                                  │
│                                                                             │
│  7. 损失计算                                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  主损失: task_loss(participant_repr, participant_y)                 │   │
│  │  辅助损失1: task_loss(session_reprs[valid], session_targets)       │   │
│  │  辅助损失2: cross_entropy(session_type_logits, session_types)      │   │
│  │  总损失: main + w1*session + w2*type                                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 与单会话数据集的区别

| 特性 | MultimodalDataset | GroupedParticipantDataset |
|------|-------------------|---------------------------|
| 粒度 | 会话级 | 参与者级 |
| 样本 | 一个会话 | 4个会话（可能部分缺失） |
| 标签 | 会话级标签 | 参与者级标签 |
| 输出 | 会话级预测 | 参与者级预测（聚合4个会话） |
| 辅助任务 | 无 | 会话类型分类 |
| 适用场景 | 简单场景 | 复杂场景、多会话融合 |

## 使用示例

```python
from common.data.grouped_dataset import GroupedParticipantDataset, grouped_collate_fn
from common.data.dataset import FeatureConfig
from torch.utils.data import DataLoader

# 配置
cfg = FeatureConfig(
    feature_root="/path/to/features",
    audio_features=["mel_mfcc", "vad", "ssl_embed"],
    video_features=["headpose_geom", "face_behavior", "vision_ssl_embed"],
    mask_policy="and_core",
)

# 创建训练集（启用会话丢弃）
train_ds = GroupedParticipantDataset(
    "manifests/train.csv",
    cfg,
    split="train",
    session_drop_prob=0.1,  # 10%概率丢弃会话
)

# 创建验证集（不丢弃）
val_ds = GroupedParticipantDataset(
    "manifests/val.csv",
    cfg,
    split="val",
    session_drop_prob=0.0,
)

# 预加载
train_ds.preload(desc="Preload train")
val_ds.preload(desc="Preload val")

# 数据加载器
train_loader = DataLoader(
    train_ds,
    batch_size=16,
    shuffle=True,
    collate_fn=grouped_collate_fn,
    num_workers=0,  # 预加载后设为0
    pin_memory=True,
)

# 训练循环
for batch in train_loader:
    flat_batch = batch["flat_batch"]
    session_valid = batch["session_valid"]  # (B, 4)
    y_a1 = batch["participant_y_a1"]         # (B, 3)
    n_participants = batch["n_participants"]  # B
    
    # 模型前向
    out = grouped_model(flat_batch, n_participants, session_valid)
    participant_repr = out["participant_repr"]  # (B, d_shared)
    session_reprs = out["session_reprs"]        # (B×4, d_shared)
    
    # 任务预测
    logits = task_head(participant_repr)  # (B, 3) for A1
    loss = a1_loss(logits, y_a1)
    ...
```