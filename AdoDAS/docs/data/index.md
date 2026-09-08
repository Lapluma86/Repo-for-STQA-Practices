# 数据模块文档

## 目录

1. [feature_io.py](#feature_io) - 特征文件I/O操作
2. [dataset.py](#dataset) - 单会话数据集处理
3. [grouped_dataset.py](#grouped_dataset) - 多会话分组数据集处理

---

## feature_io.py

### 概述

`feature_io.py` 负责从磁盘加载原始的音频和视频特征文件。它提供了统一的接口来读取各种格式的特征数据,包括时序序列特征(.npz)和池化特征(.parquet/.json)。

### 核心数据结构

```python
class SequenceData(NamedTuple):
    features: np.ndarray      # 特征矩阵 (T, D)
    timestamps_ms: np.ndarray # 时间戳数组 (T,)
    valid_mask: np.ndarray    # 有效帧掩码 (T,)
```

### 主要函数

#### load_sequence

```python
def load_sequence(
    root: Path,
    split: str,
    anon_school: str,
    anon_class: str,
    anon_pid: str,
    modality: str,          # "audio" 或 "video"
    feature_set: str,       # 特征名称,如 "mel_mfcc"
    session: str,           # 会话ID,如 "A01"
    model_tag: str | None = None,  # SSL模型标签
) -> SequenceData
```

**功能**: 从指定的路径加载时序特征序列。

**文件路径结构**:
```
root/split/school/class/pid/modality/feature_set/[model_tag/]session/sequence.npz
```

**支持的特征类型**:

| 特征名 | 特殊处理 | 说明 |
|--------|----------|------|
| mel_mfcc | 合并mel_features和mfcc_features | 梅尔频谱 + MFCC |
| 其他 | 直接读取features键 | 如vad, ssl_embed等 |

**返回值**:
- `features`: 形状为 (T, D) 的浮点数组,其中T是时间帧数,D是特征维度
- `timestamps_ms`: 每帧的时间戳(毫秒)
- `valid_mask`: 每帧的有效性布尔掩码

**异常**:
- `FileNotFoundError`: 序列文件不存在
- `KeyError`: 特征文件缺少必要键

#### load_egemaps_pooled

```python
def load_egemaps_pooled(
    root: Path,
    split: str,
    anon_school: str,
    anon_class: str,
    anon_pid: str,
    session: str,
) -> np.ndarray | None
```

**功能**: 加载池化的egemaps特征。

**文件路径**:
```
.../audio/egemaps/session/pooled.parquet
.../audio/egemaps/session/pooled.json
```

**特点**:
- 尝试parquet格式(优先)和json格式
- 返回一维数组(无时间维度)
- 失败时返回None

### 辅助函数

#### discover_feature_sets

```python
def discover_feature_sets(
    root: Path, 
    split: str, 
    modality: str, 
    limit: int = 5
) -> dict[str, list[str]]
```

自动发现数据集中可用的特征集和SSL模型标签。

#### list_file_ids

```python
def list_file_ids(root: Path, split: str, limit: int = 0) -> list[tuple[str, str, str]]
```

列出split目录下的所有参与者(school, class, pid)。

---

## dataset.py

### 概述

`dataset.py` 提供基础的数据集处理功能,支持单会话级别的特征加载、对齐和掩码计算。这是 `grouped_dataset.py` 的基础。

### 核心配置类

```python
@dataclass
class FeatureConfig:
    feature_root: str                  # 特征根目录
    audio_features: list[str]          # 使用的音频特征列表
    video_features: list[str]          # 使用的视频特征列表
    audio_ssl_model_tag: str           # 音频SSL模型标签
    video_ssl_model_tag: str           # 视频SSL模型标签
    grid_step_ms: float = 40.0         # 对齐网格步长(毫秒)
    tolerance_ms: float = 25.0         # 对齐容差(毫秒)
    mask_policy: str = "and_core"      # 掩码策略
    core_audio: list[str]              # 核心音频特征
    core_video: list[str]              # 核心视频特征
```

### 关键算法: align_to_grid

```python
def align_to_grid(
    groups: dict[str, SequenceData],
    grid_step_ms: float = 40.0,
    tolerance_ms: float = 25.0,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, int]
```

**功能**: 将不同采样率/时间戳的特征对齐到统一的时间网格。

**算法步骤**:

1. **确定时间范围**:
   ```python
   t_min = min(seq.timestamps_ms[0] for seq in groups.values())
   t_max = max(seq.timestamps_ms[-1] for seq in groups.values())
   grid = np.arange(t_min, t_max + grid_step_ms * 0.5, grid_step_ms)
   ```

2. **对每个特征组执行近邻插值**:
   ```python
   # 1. 二分查找最近的时间索引
   idx = np.searchsorted(timestamps, grid, side="left")
   idx = np.clip(idx, 0, len(timestamps) - 1)
   
   # 2. 比较左右距离,选择更近的
   idx_left = np.clip(idx - 1, 0, len(timestamps) - 1)
   dist_right = np.abs(grid - timestamps[idx])
   dist_left = np.abs(grid - timestamps[idx_left])
   use_left = dist_left < dist_right
   best_idx = np.where(use_left, idx_left, idx)
   best_dist = np.where(use_left, dist_left, dist_right)
   
   # 3. 检查是否在容差范围内
   within = best_dist <= tolerance_ms
   ```

3. **输出对齐后的特征**:
   - 特征形状: (T, D), 其中T是网格点数
   - 掩码形状: (T,), 表示每个时间点的有效性

**为什么需要对齐?**

不同特征可能有不同的采样率:
- mel_mfcc: 通常10ms一帧
- ssl_embed: 通常20ms一帧
- face_behavior: 通常33ms一帧

对齐到统一网格(默认40ms)确保所有特征在同一时间点有值。

### 掩码策略

```python
def _compute_modality_mask(
    mask_parts: list[np.ndarray],  # 各特征的掩码
    mask_names: list[str],         # 特征名称
    core_names: list[str],         # 核心特征
    policy: str,                   # 策略
    T: int,
) -> np.ndarray
```

**三种策略**:

1. **"or"**: 任一特征有效即有效
   ```python
   return np.any(np.stack(mask_parts), axis=0)
   ```

2. **"and_core"**: 所有核心特征有效才有效
   ```python
   core_masks = [m for m, n in zip(mask_parts, mask_names) if n in core_names]
   if core_masks:
       return np.all(np.stack(core_masks), axis=0)
   return np.any(np.stack(mask_parts), axis=0)
   ```

3. **"require_k"**: 至少K个核心特征有效
   ```python
   k = max(1, len(core_names))
   stacked = np.stack(mask_parts)
   return np.sum(stacked, axis=0) >= k
   ```

**为什么需要掩码策略?**

某些特征可能在特定时间段缺失:
- 音频SSL嵌入可能在静音时缺失
- 视频特征可能在人脸检测失败时缺失
- 掩码确保模型只在有效数据上计算

### MultimodalDataset 类

```python
class MultimodalDataset(Dataset):
    def __init__(self, manifest_path, cfg, split)
    def __getitem__(self, idx) -> dict[str, Any]
```

**返回样本格式**:

```python
{
    # 时序特征组 (T, D)
    "audio_groups": {
        "mel_mfcc": Tensor(T, D_mel),      # 梅尔频谱+MFCC
        "vad": Tensor(T, 1),                # 语音活动检测
        "ssl_embed": Tensor(T, D_ssl),      # SSL语音嵌入
    },
    "video_groups": {
        "headpose_geom": Tensor(T, D_pose), # 头部姿态几何
        "face_behavior": Tensor(T, D_beh),  # 面部行为特征
        # ...
    },
    
    # 池化特征 (D,)
    "audio_pooled_groups": {
        "egemaps": Tensor(D_egemaps),       # egemaps统计特征
    },
    
    # 掩码和信号
    "mask_audio": Tensor(T, bool),          # 音频有效性掩码
    "mask_video": Tensor(T, bool),          # 视频有效性掩码
    "vad_signal": Tensor(T, float32),       # VAD信号(用于ASP)
    "qc_quality": Tensor(T, float32),       # 质量控制信号
    
    # 元数据
    "session_idx": int,                      # 会话类型索引(0-3)
    "y_a1": Tensor(3, float32),              # A1标签(D, A, S)
    "y_a2": Tensor(21, float32),             # A2标签(21个项目)
    "seq_len": int,                          # 序列长度T
    "anon_pid": str,                         # 参与者ID
    "session": str,                          # 会话ID
}
```

### collate_fn 批处理

```python
def collate_fn(batch: list[dict]) -> dict[str, Any]
```

**功能**: 将多个样本整理为一个批次。

**处理步骤**:

1. **确定批次最大长度**: `T_max = max(b["seq_len"] for b in batch)`

2. **填充时序特征**:
   ```python
   # 创建 (B, T_max, D) 的张量
   t = torch.zeros(B, T_max, D)
   for i, b in enumerate(batch):
       L = b["seq_len"]
       t[i, :L] = b[key][n]
   ```

3. **创建填充掩码**:
   ```python
   pad_mask = torch.ones(B, T_max, dtype=torch.bool)
   for i, b in enumerate(batch):
       pad_mask[i, :b["seq_len"]] = False
   ```

4. **处理池化特征**:
   ```python
   # 直接堆叠,无时间维度
   torch.stack([b["audio_pooled_groups"][name] for b in batch])
   ```

5. **处理元数据**:
   ```python
   torch.stack([b["y_a1"] for b in batch])  # (B, 3)
   torch.stack([b["y_a2"] for b in batch])  # (B, 21)
   ```

---

## grouped_dataset.py

### 概述

`grouped_dataset.py` 是核心数据集类,实现"参与者级别"的数据组织。每个参与者包含4个会话(A01, B01, B02, B03),支持多会话聚合预测。

### 会话类型

```python
SESSIONS = ["A01", "B01", "B02", "B03"]
SESSION_TO_IDX = {"A01": 0, "B01": 1, "B02": 2, "B03": 3}
```

**会话内容**:
- **A01**: "北风与太阳"标准化朗读文本
- **B01**: 描述昨天过得怎么样
- **B02**: 描述最近一周最开心的记忆
- **B03**: 描述最近一周最悲伤的记忆

### GroupedParticipantDataset 类

```python
class GroupedParticipantDataset(Dataset):
    def __init__(
        self,
        manifest_path,
        cfg: FeatureConfig,
        split: str,
        session_drop_prob: float = 0.0,  # 仅训练集
    )
```

**初始化流程**:

1. **按参与者分组**:
   ```python
   group_cols = ["anon_school", "anon_class", "anon_pid"]
   grouped = manifest.groupby(group_cols)
   
   for (school, cls, pid), group in grouped:
       # 收集该参与者的所有会话
       sess_rows = {}
       for _, row in group.iterrows():
           sess_rows[str(row["session"])] = row
       
       # 创建参与者记录
       self.participants.append({
           "anon_school": str(school),
           "anon_class": str(cls),
           "anon_pid": str(pid),
           "sess_rows": sess_rows,
           "y_a1": ...,  # 从任意行读取标签
           "y_a2": ...,  # 从任意行读取标签
       })
   ```

2. **标签说明**:
   - 所有会话共享相同的标签(因为是参与者级别的任务)
   - 从 `any_row = group.iloc[0]` 读取标签

### 会话丢弃 (Session Dropout)

```python
def _apply_session_dropout(self, sample: dict) -> dict:
    """训练时随机丢弃部分会话,增强泛化"""
    valid_indices = [
        idx for idx, is_valid in enumerate(sample["session_valid"].tolist())
        if is_valid
    ]
    if len(valid_indices) <= 1:
        return sample  # 至少保留一个
    
    if np.random.random() < self.session_drop_prob:
        drop_idx = int(np.random.choice(valid_indices))
        # 标记该会话为无效
        sessions[drop_idx] = None
        session_valid[drop_idx] = False
```

**为什么需要会话丢弃?**

- 测试时参与者可能只有部分会话数据
- 强制模型不过度依赖特定会话
- 增加训练样本多样性
- 提高对缺失数据的鲁棒性

### 数据加载流程

```python
def __getitem__(self, idx) -> dict[str, Any]:
    # 1. 从缓存或磁盘加载参与者
    if self._cache is not None:
        sample = self._cache[idx]
    else:
        sample = self._load_participant(idx)
    
    # 2. 应用会话丢弃(训练时)
    if self.split == "train" and self.session_drop_prob > 0.0:
        return self._apply_session_dropout(sample)
    return sample
```

**加载单个参与者**:

```python
def _load_participant(self, idx: int) -> dict:
    info = self.participants[idx]
    sessions_data = []
    session_valid = []
    
    # 加载4个会话,缺失的填充None
    for sess_name in SESSIONS:
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
        "sessions": sessions_data,           # 4个会话数据
        "session_valid": np.array(session_valid),  # (4,) bool
        "y_a1": torch.from_numpy(info["y_a1"]),    # (3,)
        "y_a2": torch.from_numpy(info["y_a2"]),    # (21,)
        "anon_pid": info["anon_pid"],
        "anon_school": info["anon_school"],
        "anon_class": info["anon_class"],
        "session_names": SESSIONS,
    }
```

### grouped_collate_fn 批处理

```python
def grouped_collate_fn(batch: list[dict]) -> dict[str, Any]:
    """整理参与者批次,展平所有会话"""
    B = len(batch)  # 批次参与者数
    
    # 1. 收集所有有效会话
    all_sessions = []
    session_types = []
    session_valid_list = []
    flat_pids = []
    flat_sess_names = []
    
    for b_idx, sample in enumerate(batch):
        session_valid_list.append(sample["session_valid"])  # (4,)
        
        for s_idx, sess_data in enumerate(sample["sessions"]):
            if sess_data is not None:
                all_sessions.append(sess_data)
                session_types.append(s_idx)  # 会话类型(0-3)
                flat_pids.append(sample["anon_pid"])
                flat_sess_names.append(SESSIONS[s_idx])
            else:
                # 创建虚拟会话填充
                dummy = _make_dummy_session(ref)
                all_sessions.append(dummy)
                session_types.append(s_idx)
    
    n_flat = len(all_sessions)  # = B × 4
    
    # 2. 填充到统一长度
    T_max = max(s["seq_len"] for s in all_sessions)
    
    # 返回结构:
    return {
        "flat_batch": {  # 展平批次,用于Backbone
            "audio_groups": {...},      # (n_flat, T_max, D_audio)
            "video_groups": {...},      # (n_flat, T_max, D_video)
            "pad_mask": ...,            # (n_flat, T_max)
            # ...
        },
        "participant_y_a1": ...,        # (B, 3)
        "participant_y_a2": ...,        # (B, 21)
        "session_valid": ...,           # (B, 4) - 哪些会话有效
        "session_types": ...,           # (n_flat,) - 每个样本的会话类型
        "n_participants": B,
        "anon_pids": [...],             # B个参与者ID
        "flat_sessions": [...],         # n_flat个会话ID
    }
```

**为什么需要展平?**

- Backbone处理单个会话,期望输入形状 (N, T, D)
- 4个参与者的4个会话 = 16个"独立"样本
- Backbone并行处理这16个样本
- 之后在GroupedModel中重新组织回(B, 4, D)

### 虚拟会话 (Dummy Session)

```python
def _make_dummy_session(ref: dict) -> dict:
    """创建零填充的虚拟会话,用于缺失会话"""
    T = 1  # 最小长度
    return {
        "audio_groups": {k: zeros(T, D) for k, v in ref["audio_groups"].items()},
        "video_groups": {k: zeros(T, D) for k, v in ref["video_groups"].items()},
        "mask_audio": zeros(T, bool),
        "mask_video": zeros(T, bool),
        "seq_len": T,
        # ...
    }
```

**为什么需要虚拟会话?**

- PyTorch DataLoader需要固定大小的批次
- 不同参与者可能有不同数量的有效会话(0-4)
- 虚拟会话保持张量形状一致
- 掩码(mask)确保模型忽略虚拟数据

### 预加载机制

```python
def preload(self, desc: str = None) -> float:
    """将所有数据预加载到RAM,加快训练"""
    self._cache = [None] * len(self)
    for i in tqdm(range(len(self))):
        try:
            self._cache[i] = self._load_participant(i)
        except Exception as exc:
            log.warning(f"Preload failed for participant {i}: {exc}")
    
    # 估算内存占用
    gb = self._estimate_cache_bytes() / 1024**3
    log.info(f"Preloaded {len(self)} participants ({gb:.1f} GB)")
    return gb
```

**优点**:
- 避免训练时的磁盘I/O瓶颈
- 特别适合多次迭代的训练
- 数据加载器可以设置 `num_workers=0`, 避免多进程开销

**缺点**:
- 占用大量RAM(可能数十GB)
- 首次加载时间较长

### 配置示例

```yaml
# tasks/a2/default.yaml 中的数据配置部分
feature_root: "/data1/AdoDas"
manifest_dir: "/data1/AdoDas"

feature_selection:
  audio_features:
    - mel_mfcc
    - vad
    - ssl_embed
    - egemaps
  video_features:
    - headpose_geom
    - face_behavior
    - qc_stats
    - vad_agg
    - body_pose
    - global_motion
    - vision_ssl_embed
  audio_ssl_model_tag: "chinese-hubert-large"
  video_ssl_model_tag: "vit-mae-base"

mask_policy: "and_core"
core_audio: ["mel_mfcc", "ssl_embed"]
core_video: ["vision_ssl_embed", "qc_stats"]

session_drop_prob: 0.1
feature_noise_std: 0.01
label_smoothing: 0.05
preload: true
num_workers: 8
```

### 数据目录结构与路径解析

```
/data1/AdoDas/
├── Train/train.csv              # 训练集 manifest (16801 行)
├── Train/train/train/           # 训练集特征 (SCH_xxx/CLS_xxx/P_xxx/...)
├── Val/val.csv                  # 验证集 manifest (2401 行)
├── Val/val/val/                 # 验证集特征
├── Test/test/test_hidden/       # 测试集特征 (无 manifest, 无标签)
└── output/                      # 训练输出
```

逻辑 split 名 → 实际子路径映射 (`SPLIT_DATA_PATH` in `common/data/dataset.py`):
| 逻辑 split | 子路径 | 说明 |
|-----------|--------|------|
| `train` | `Train/train/train` | 训练集特征目录 |
| `val` | `Val/val/val` | 验证集特征目录 |
| `test_hidden` | `Test/test/test_hidden` | 测试集特征目录 |

数据集加载流程: `feature_root / SPLIT_DATA_PATH[split] / school / class / pid / modality / feature / session / sequence.npz`

特征维度（实测）:
| 特征 | 维度 | 类型 |
|------|------|------|
| mel_mfcc | 93 | 音频时序 |
| vad | 1 | 音频时序 |
| ssl_embed (chinese-hubert-large) | 1024 | 音频时序 |
| egemaps | 88 | 音频池化 |
| headpose_geom | 5 | 视频时序 |
| face_behavior | 5 | 视频时序 |
| qc_stats | 4 | 视频时序 |
| vad_agg | 4 | 视频时序 |
| body_pose | 27 | 视频时序 |
| global_motion | 4 | 视频时序 |
| vision_ssl_embed (vit-mae-base) | 768 | 视频时序 |

### HDF5 加速

将分散的 npz 文件打包为单文件 HDF5，减少小文件 I/O 和 NFS 元数据开销。

```bash
# 全量打包
python scripts/pack_features.py \
  --manifest /data1/AdoDas/Train/train.csv \
  --feature-root /data1/AdoDas \
  --split train \
  --output /data1/AdoDas/train_packed.h5

# Debug 子集
python scripts/pack_features.py \
  --manifest /data1/AdoDas/Train/train.csv \
  --feature-root /data1/AdoDas \
  --split train \
  --output /data1/AdoDas/train_debug.h5 \
  --max-participants 100
```

打包后在 YAML 中设置 `use_hdf5: true`。数据集自动 switch 到 `HDF5GroupedDataset`，加载速度提升 3-5x。

---

## 数据模块设计要点

### 1. 分层设计

```
feature_io.py  →  底层I/O,文件格式抽象
     ↓
dataset.py     →  单会话处理,时序对齐
     ↓
grouped_dataset.py → 多会话聚合,参与者级
```

### 2. 灵活性

- **特征可配置**: 通过FeatureConfig选择使用哪些特征
- **模态可扩展**: 容易添加新的音频/视频特征
- **策略可切换**: 掩码策略、聚合方法可配置

### 3. 效率考虑

- **延迟加载**: 只在需要时从磁盘读取
- **内存缓存**: 可选预加载到RAM
- **批处理优化**: grouped_collate_fn一次处理多个参与者

### 4. 数据完整性

- **掩码机制**: 明确标记无效数据点
- **缺失处理**: 虚拟会话+掩码处理缺失数据
- **对齐验证**: 严格检查时间戳和特征维度
