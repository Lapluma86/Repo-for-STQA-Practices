# dataset.py - 单会话数据集处理详解

## 文件概述

`dataset.py` 定义了单会话级别的数据集处理逻辑，包括特征配置、数据对齐、有效性掩码计算等核心功能。它是 `grouped_dataset.py` 的基础模块。

## 常量定义

```python
# 四种会话类型
SESSIONS = ["A01", "B01", "B02", "B03"]
SESSION_TO_IDX = {s: i for i, s in enumerate(SESSIONS)}  # {"A01": 0, "B01": 1, ...}

# A2任务的21个评估项目列名
ITEM_COLS = [f"d{i:02d}" for i in range(1, 22)]  # ["d01", "d02", ..., "d21"]

# A1任务的三个标签列名
A1_COLS = ["y_D", "y_A", "y_S"]  # 抑郁、焦虑、压力

# 池化音频特征集合
POOLED_AUDIO_FEATURES = {"egemaps"}
```

### 会话类型说明

| 会话 | 类型 | 内容 |
|------|------|------|
| A01 | 朗读 | "北风与太阳"标准化阅读段落 |
| B01 | 日常 | 描述昨天过得怎么样 |
| B02 | 开心 | 描述最近一周最开心的记忆 |
| B03 | 悲伤 | 描述最近一周最悲伤的记忆 |

## FeatureConfig 数据类

```python
@dataclass
class FeatureConfig:
    # 特征根目录路径
    feature_root: str = "/media/k3nwong/Data1/test/outputs/pipeline/anonymized"
    
    # 音频特征列表
    audio_features: list[str] = field(
        default_factory=lambda: ["mel_mfcc", "vad", "egemaps", "ssl_embed"]
    )
    
    # 视频特征列表
    video_features: list[str] = field(
        default_factory=lambda: [
            "headpose_geom", "face_behavior", "qc_stats", "vad_agg",
            "body_pose", "global_motion", "vision_ssl_embed",
        ]
    )
    
    # SSL模型标签
    audio_ssl_model_tag: str = "chinese-hubert-base"
    video_ssl_model_tag: str = "dinov2-base"
    
    # 数据对齐参数
    grid_step_ms: float = 40.0      # 网格步长（毫秒）
    tolerance_ms: float = 25.0      # 时间容差（毫秒）
    
    # 有效性掩码策略
    mask_policy: str = "and_core"  # 掩码计算策略
    
    # 核心特征（必须存在的）
    core_audio: list[str] = field(default_factory=lambda: ["mel_mfcc", "vad"])
    core_video: list[str] = field(default_factory=lambda: ["face_behavior", "headpose_geom"])
```

### 特征类型属性

```python
@property
def audio_sequence_features(self) -> list[str]:
    """音频时序特征（非池化）"""
    return [name for name in self.audio_features if name not in POOLED_AUDIO_FEATURES]

@property
def audio_pooled_features(self) -> list[str]:
    """音频池化特征"""
    return [name for name in self.audio_features if name in POOLED_AUDIO_FEATURES]
```

## 数据对齐核心函数

### align_to_grid() - 特征时间对齐

```python
def align_to_grid(
    groups: dict[str, SequenceData],  # 多个特征组
    grid_step_ms: float = 40.0,       # 网格步长
    tolerance_ms: float = 25.0,       # 容差阈值
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, int]:
    """
    返回:
        aligned_feats: 对齐后的特征字典
        aligned_masks: 对齐后的掩码字典
        grid: 时间网格数组
        T: 网格长度
    """
```

**对齐原理详解**：

1. **创建时间网格**：
   ```python
   # 找到所有特征的最小和最大时间戳
   t_min = min(seq.timestamps_ms[0] for seq in groups.values())
   t_max = max(seq.timestamps_ms[-1] for seq in groups.values())
   
   # 创建均匀网格
   grid = np.arange(t_min, t_max + grid_step_ms * 0.5, grid_step_ms)
   ```
   
   示例：假设 t_min=0ms, t_max=1000ms, grid_step=40ms
   → grid = [0, 40, 80, 120, ..., 1000] (共26个点)

2. **最近邻插值**：
   ```python
   def _nearest_indices(grid: np.ndarray, timestamps: np.ndarray):
       # 使用二分搜索找到每个网格点的最近时间戳索引
       idx = np.searchsorted(timestamps, grid, side="left")
       idx = np.clip(idx, 0, len(timestamps) - 1)
       
       # 比较左右两个候选，选择距离更近的
       idx_left = np.clip(idx - 1, 0, len(timestamps) - 1)
       dist_right = np.abs(grid - timestamps[idx])
       dist_left = np.abs(grid - timestamps[idx_left])
       use_left = dist_left < dist_right
       
       best_idx = np.where(use_left, idx_left, idx)
       best_dist = np.where(use_left, dist_left, dist_right)
       return best_idx, best_dist
   ```

3. **有效性过滤**：
   ```python
   # 只接受距离在容差范围内的插值
   within = best_dist <= tolerance_ms
   aligned_feats[name] = seq.features[best_idx]
   aligned_masks[name] = seq.valid_mask[best_idx] & within
   ```

**为什么需要数据对齐？**

- 不同特征可能有不同的采样率和时间戳
- 例如：音频可能每20ms一帧，视频可能每33ms一帧
- 对齐到统一网格便于后续处理和融合

### 对齐流程图

```
原始时间戳（不同特征可能不同步）:
┌─────────────────────────────────────────────────────────────────┐
│  Feature A: timestamps = [0, 20, 40, 60, 80, 100, ...]          │
│  Feature B: timestamps = [0, 33, 66, 99, 132, ...]              │
│  Feature C: timestamps = [5, 45, 85, 125, ...]                  │
└─────────────────────────────────────────────────────────────────┘
                          ↓
创建统一时间网格:
┌─────────────────────────────────────────────────────────────────┐
│  Grid: [0, 40, 80, 120, 160, 200, ...]  (步长40ms)               │
└─────────────────────────────────────────────────────────────────┘
                          ↓
最近邻插值 + 有效性检查:
┌─────────────────────────────────────────────────────────────────┐
│  Grid[0]=0:   A→0(近), B→0(近), C→5(容差内) ✓                    │
│  Grid[40]=40: A→40(近), B→33(容差内), C→45(容差内) ✓             │
│  Grid[80]=80: A→80(近), B→66(容差内), C→85(容差内) ✓             │
│  Grid[120]:   A→100(容差外), B→99(容差内), C→125(容差外)         │
│              → 只有B有效                                        │
└─────────────────────────────────────────────────────────────────┘
                          ↓
对齐结果:
┌─────────────────────────────────────────────────────────────────┐
│  aligned_feats["A"] = [feat0, feat1, feat2, feat100...]         │
│  aligned_masks["A"] = [True,  True,  True,  False...]           │
│  aligned_feats["B"] = [feat0, feat33, feat66, feat99...]        │
│  aligned_masks["B"] = [True,  True,  True,  True...]            │
│  aligned_feats["C"] = [feat5,  feat45, feat85, None...]         │
│  aligned_masks["C"] = [True,  True,  True,  False...]           │
└─────────────────────────────────────────────────────────────────┘
```

## 有效性掩码计算

### _compute_modality_mask() - 模态掩码策略

```python
@staticmethod
def _compute_modality_mask(
    mask_parts: list[np.ndarray],   # 各特征的掩码
    mask_names: list[str],          # 特征名称
    core_names: list[str],          # 核心特征名称
    policy: str,                    # 策略名称
    T: int,                         # 时间长度
) -> np.ndarray:
```

**三种掩码策略详解**：

1. **"or" 策略**（宽松）：
   ```python
   # 任意一个特征有效，该帧就有效
   return np.any(np.stack(mask_parts), axis=0)
   ```
   
   适用场景：容忍部分特征缺失，最大化数据利用

2. **"and_core" 策略**（推荐）：
   ```python
   # 所有核心特征必须同时有效
   core_masks = [m for m, n in zip(mask_parts, mask_names) if n in core_names]
   if core_masks:
       return np.all(np.stack(core_masks), axis=0)
   return np.any(np.stack(mask_parts), axis=0)
   ```
   
   适用场景：确保核心特征（如 mel_mfcc, vad）同步，保证数据质量

3. **"require_k" 策略**（中等）：
   ```python
   # 至少 k 个核心特征有效
   k = max(1, len(core_names))
   core_masks = [m for m, n in zip(mask_parts, mask_names) if n in core_names]
   if core_masks:
       return np.sum(np.stack(core_masks), axis=0) >= k
   return np.any(np.stack(mask_parts), axis=0)
   ```
   
   适用场景：允许部分核心特征缺失，平衡质量和利用率

**掩码策略对比图**：

```
帧有效性示例 (假设 core_audio = ["mel_mfcc", "vad"]):
┌───────────────────────────────────────────────────────────────────┐
│  帧:           0     1     2     3     4     5                     │
│  mel_mfcc:    [1]   [1]   [0]   [1]   [1]   [0]                    │
│  vad:         [1]   [1]   [1]   [0]   [1]   [1]                    │
│  ssl_embed:   [1]   [0]   [1]   [1]   [0]   [1]                    │
├───────────────────────────────────────────────────────────────────┤
│  策略 "or":   [1]   [1]   [1]   [1]   [1]   [1]  ← 最宽松          │
│  策略 "and_core": [1]   [1]   [0]   [0]   [1]   [0]  ← 最严格      │
│  策略 "require_k": [1]   [1]   [1]   [1]   [1]   [1]  ← 中等       │
│                  (至少1个核心特征有效)                             │
└───────────────────────────────────────────────────────────────────┘
```

## MultimodalDataset 数据集类

### 初始化

```python
class MultimodalDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,  # manifest CSV文件路径
        cfg: FeatureConfig,          # 特征配置
        split: str,                  # 数据集划分
    ):
```

**Manifest CSV 结构**：
```
anon_school, anon_class, anon_pid, session, y_D, y_A, y_S, d01, d02, ...
s001,        c001,       p001,     A01,     1,   0,   1,   2,   1,   ...
s001,        c001,       p001,     B01,     1,   0,   1,   2,   1,   ...
...
```

**必需列检查**：
```python
required = {"anon_school", "anon_class", "anon_pid", "session"}
missing = required - set(self.manifest.columns)
if missing:
    raise KeyError(f"Manifest missing columns: {missing}")
```

### 特征维度探测

```python
@property
def feature_dims(self) -> dict[str, int]:
    """懒加载计算特征维度"""
    if self._feature_dims is None:
        self._feature_dims = self._probe_dims()
    return self._feature_dims

def _probe_dims(self) -> dict[str, int]:
    """从第一个样本探测维度"""
    row = self.manifest.iloc[0]
    dims = {}
    
    # 探测音频特征维度
    for name, seq in self._load_raw_groups(row, "audio").items():
        dims[name] = seq.features.shape[1]
    
    # 探测视频特征维度
    for name, seq in self._load_raw_groups(row, "video").items():
        dims[name] = seq.features.shape[1]
    
    # 探测池化特征维度
    if "egemaps" in self.cfg.audio_pooled_features:
        eg = load_egemaps_pooled(...)
        if eg is not None:
            dims["egemaps"] = len(eg)
    
    return dims
```

**为什么需要维度探测？**
- 不同特征集可能有不同维度
- 模型初始化需要知道输入维度
- 懒加载避免重复计算

### 数据预加载

```python
def preload(self, desc: str | None = None) -> float:
    """预加载所有数据到内存"""
    n = len(self)
    self._cache = [None] * n
    errors = 0
    
    for i in tqdm(range(n), desc=desc):
        try:
            self._cache[i] = self._load_sample(i)
        except Exception as exc:
            errors += 1
            if errors <= 3:
                log.warning(f"Preload: sample {i} failed: {exc}")
    
    # 计算内存占用
    gb = self._estimate_cache_bytes() / 1024**3
    log.info(f"Preloaded {n - errors}/{n} samples ({gb:.1f} GB in RAM)")
    return gb
```

**预加载的好处**：
- 减少训练时的IO延迟
- 加快数据访问速度
- 特别是对于小数据集或内存充足的情况

**内存估算**：
```python
def _estimate_cache_bytes(self) -> int:
    total = 0
    for sample in self._cache:
        if sample is None:
            continue
        for v in sample.values():
            if isinstance(v, torch.Tensor):
                total += v.nelement() * v.element_size()
            elif isinstance(v, dict):
                for vv in v.values():
                    if isinstance(vv, torch.Tensor):
                        total += vv.nelement() * vv.element_size()
    return total
```

### 样本加载流程

```python
def _load_sample(self, idx: int) -> dict[str, Any]:
    row = self.manifest.iloc[idx]
    
    # 1. 加载原始特征
    audio_raw = self._load_raw_groups(row, "audio")
    video_raw = self._load_raw_groups(row, "video")
    
    # 2. 合并所有特征组
    all_groups = {}
    for k, v in audio_raw.items():
        all_groups[f"audio/{k}"] = v
    for k, v in video_raw.items():
        all_groups[f"video/{k}"] = v
    
    # 3. 对齐到统一网格
    aligned_feats, aligned_masks, grid_ms, T = align_to_grid(all_groups, ...)
    
    # 4. 分离音频和视频特征
    audio_groups = {}
    video_groups = {}
    for key, feat in aligned_feats.items():
        modality, name = key.split("/", 1)
        if modality == "audio":
            audio_groups[name] = torch.from_numpy(feat.astype(np.float32))
        else:
            video_groups[name] = torch.from_numpy(feat.astype(np.float32))
    
    # 5. 计算模态掩码
    mask_audio = self._compute_modality_mask(audio_mask_parts, ..., cfg.mask_policy, T)
    mask_video = self._compute_modality_mask(video_mask_parts, ..., cfg.mask_policy, T)
    
    # 6. 提取VAD和QC信号
    vad_signal = aligned_feats["audio/vad"][:, 0] * aligned_masks["audio/vad"]
    qc_quality = aligned_feats["video/qc_stats"][:, 0] * aligned_masks["video/qc_stats"]
    
    # 7. 加载池化特征
    audio_pooled_groups = {}
    if "egemaps" in cfg.audio_pooled_features:
        egemaps = load_egemaps_pooled(...)
        audio_pooled_groups["egemaps"] = torch.from_numpy(egemaps)
    
    # 8. 填充缺失特征
    dims = self.feature_dims
    for name in cfg.audio_features:
        if name not in audio_groups and name in dims:
            audio_groups[name] = torch.zeros(T, dims[name])
    
    # 9. 提取标签
    y_a1 = np.array([float(row.get(c, -1)) for c in A1_COLS], dtype=np.float32)
    y_a2 = np.array([float(row.get(c, -1)) for c in ITEM_COLS], dtype=np.float32)
    
    # 10. 返回完整样本
    return {
        "audio_groups": audio_groups,
        "audio_pooled_groups": audio_pooled_groups,
        "video_groups": video_groups,
        "mask_audio": torch.from_numpy(mask_audio),
        "mask_video": torch.from_numpy(mask_video),
        "vad_signal": torch.from_numpy(vad_signal),
        "qc_quality": torch.from_numpy(qc_quality),
        "session_idx": SESSION_TO_IDX.get(str(row["session"]), 0),
        "y_a1": torch.from_numpy(y_a1),
        "y_a2": torch.from_numpy(y_a2),
        "seq_len": T,
        "anon_pid": str(row["anon_pid"]),
        "session": str(row["session"]),
    }
```

### collate_fn - 批次整理函数

```python
def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    B = len(batch)
    T_max = max(b["seq_len"] for b in batch)  # 最大序列长度
```

**填充逻辑**：

```python
def _pad_groups(names: list[str], key: str) -> dict[str, torch.Tensor]:
    """填充特征组到最大长度"""
    result = {}
    for n in names:
        D = batch[0][key][n].shape[-1]  # 特征维度
        t = torch.zeros(B, T_max, D)    # 创建填充张量
        for i, b in enumerate(batch):
            L = b["seq_len"]
            t[i, :L] = b[key][n]         # 填充实际数据
        result[n] = t
    return result
```

**填充掩码**：
```python
pad_mask = torch.ones(B, T_max, dtype=torch.bool)  # 初始全为True（表示填充）
for i, b in enumerate(batch):
    pad_mask[i, :b["seq_len"]] = False  # 实际数据位置设为False
```

**返回的批次结构**：
```python
return {
    "audio_groups": {...},           # 填充后的音频特征组
    "audio_pooled_groups": {...},    # 池化音频特征
    "video_groups": {...},           # 填充后的视频特征组
    "mask_audio": (B, T_max),        # 音频有效性掩码
    "mask_video": (B, T_max),        # 视频有效性掩码
    "pad_mask": (B, T_max),          # 填充掩码
    "vad_signal": (B, T_max),        # VAD信号
    "qc_quality": (B, T_max),        # QC信号
    "session_idx": (B,),             # 会话类型索引
    "y_a1": (B, 3),                  # A1标签
    "y_a2": (B, 21),                 # A2标签
    "seq_len": (B,),                 # 序列长度
    "anon_pid": list[str],           # 参与者ID列表
    "session": list[str],            # 会话名称列表
}
```

## 数据处理完整流程图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           数据处理完整流程                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. Manifest CSV                                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  包含: anon_school, anon_class, anon_pid, session, y_D, y_A, y_S    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                          ↓                                                  │
│                                                                             │
│  2. 加载原始特征 (feature_io.py)                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  audio/mel_mfcc/sequence.npz                                         │   │
│  │  audio/vad/sequence.npz                                              │   │
│  │  audio/ssl_embed/chinese-hubert-base/sequence.npz                    │   │
│  │  video/headpose_geom/sequence.npz                                    │   │
│  │  ...                                                                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                          ↓                                                  │
│                                                                             │
│  3. 数据对齐 (align_to_grid)                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  创建统一时间网格 (40ms步长)                                          │   │
│  │  最近邻插值 + 容差检查 (25ms)                                         │   │
│  │  输出: aligned_feats, aligned_masks, grid, T                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                          ↓                                                  │
│                                                                             │
│  4. 特征分离和掩码计算                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  分离: audio_groups, video_groups                                    │   │
│  │  掩码: mask_audio (and_core策略), mask_video                         │   │
│  │  VAD/QC信号提取                                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                          ↓                                                  │
│                                                                             │
│  5. 池化特征加载                                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  audio/egemaps/session/pooled.parquet                                │   │
│  │  输出: audio_pooled_groups["egemaps"]                                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                          ↓                                                  │
│                                                                             │
│  6. 缺失特征填充                                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  如果某特征不存在，用零向量填充                                        │   │
│  │  保证所有批次特征结构一致                                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                          ↓                                                  │
│                                                                             │
│  7. 标签提取                                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  y_a1: [y_D, y_A, y_S] (抑郁、焦虑、压力)                             │   │
│  │  y_a2: [d01, ..., d21] (21个评估项目)                                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                          ↓                                                  │
│                                                                             │
│  8. 批次整理 (collate_fn)                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  填充到最大序列长度 T_max                                             │   │
│  │  创建填充掩码 pad_mask                                                │   │
│  │  堆叠池化特征和标签                                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                          ↓                                                  │
│                                                                             │
│  9. 模型输入                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  batch["audio_groups"]["mel_mfcc"]: (B, T_max, D_mel)                │   │
│  │  batch["video_groups"]["headpose_geom"]: (B, T_max, D_pose)          │   │
│  │  batch["mask_audio"]: (B, T_max)                                      │   │
│  │  batch["pad_mask"]: (B, T_max)                                        │   │
│  │  ...                                                                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 使用示例

```python
from common.data.dataset import FeatureConfig, MultimodalDataset, collate_fn
from torch.utils.data import DataLoader

# 配置特征
cfg = FeatureConfig(
    feature_root="/path/to/features",
    audio_features=["mel_mfcc", "vad", "ssl_embed"],
    video_features=["headpose_geom", "face_behavior"],
    audio_ssl_model_tag="chinese-hubert-base",
    mask_policy="and_core",
)

# 创建数据集
train_ds = MultimodalDataset("manifests/train.csv", cfg, split="train")

# 可选：预加载到内存
train_ds.preload(desc="Preload train")

# 创建数据加载器
train_loader = DataLoader(
    train_ds,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn,
    pin_memory=True,
)

# 训练循环
for batch in train_loader:
    # batch 包含填充后的所有特征和标签
    audio_features = batch["audio_groups"]["mel_mfcc"]  # (32, T_max, 60)
    labels = batch["y_a1"]  # (32, 3)
    ...
```