# feature_io.py - 特征文件I/O操作详解

## 文件概述

`feature_io.py` 是数据加载的基础模块，负责从磁盘读取各种类型的特征文件。它定义了特征数据的结构，并提供了统一的特征加载接口。

## 核心数据结构

### SequenceData 命名元组

```python
class SequenceData(NamedTuple):
    features: np.ndarray      # 特征矩阵，形状 (T, D)，T为时间帧数，D为特征维度
    timestamps_ms: np.ndarray # 时间戳数组，形状 (T)，单位为毫秒
    valid_mask: np.ndarray    # 有效帧掩码，形状 (T)，bool类型，标记哪些帧是有效的
```

**设计原理**：
- 使用 `NamedTuple` 而非普通类，因为它不可变、轻量、支持字段访问
- 同时包含特征、时间戳和有效掩码，确保三者同步
- 有效掩码用于标记因设备故障、遮挡等原因导致的无效帧

## 核心函数详解

### load_sequence() - 加载时序特征序列

```python
def load_sequence(
    root: Path,                # 特征根目录
    split: str,                # 数据集划分 (train/val/test_hidden)
    anon_school: str,          # 匿名化学校ID
    anon_class: str,           # 匿名化班级ID
    anon_pid: str,             # 化名化参与者ID
    modality: str,             # 模态类型 (audio/video)
    feature_set: str,          # 特征集名称 (如 mel_mfcc, vad, ssl_embed)
    session: str,              # 会话名称 (A01, B01, B02, B03)
    model_tag: str | None = None,  # SSL模型标签 (如 chinese-hubert-base)
) -> SequenceData:
```

**文件路径构建逻辑**：
```
特征文件路径 = root / split / school / class / pid / modality / feature_set / [model_tag] / session / "sequence.npz"
```

**示例路径**：
```
/root/train/school001/class001/pid001/audio/mel_mfcc/A01/sequence.npz
/root/train/school001/class001/pid001/audio/ssl_embed/chinese-hubert-base/A01/sequence.npz
/root/train/school001/class001/pid001/video/vision_ssl_embed/dinov2-base/A01/sequence.npz
```

**处理不同特征格式的逻辑**：

1. **mel_mfcc 特征**（特殊处理）：
   ```python
   # mel_mfcc 包含两个独立数组: mel_features 和 mfcc_features
   # 需要拼接成一个数组
   arrays = []
   for k in ("mel_features", "mfcc_features"):
       arrays.append(data[k].astype(np.float32))
   features = np.concatenate(arrays, axis=-1)  # 拼接最后一维
   ```

2. **其他特征**（标准处理）：
   ```python
   # 大多数特征文件只有一个 "features" 键
   features = data["features"].astype(np.float32)
   ```

**时间戳和有效掩码处理**：
```python
# 时间戳必须存在
timestamps_ms = data["timestamps_ms"].astype(np.float64)

# 有效掩码可选，默认全为True
if "valid_mask" in data:
    valid_mask = data["valid_mask"].astype(bool)
else:
    valid_mask = np.ones(len(timestamps_ms), dtype=bool)
```

**形状一致性检查**：
```python
T = len(timestamps_ms)
# 确保特征和掩码的长度与时间戳一致
if features.shape[0] != T:
    raise ValueError(f"Shape mismatch: features {features.shape[0]} vs timestamps {T}")
if valid_mask.shape[0] != T:
    raise ValueError(f"Shape mismatch: valid_mask {valid_mask.shape[0]} vs timestamps {T}")
```

### load_egemaps_pooled() - 加载池化音频特征

```python
def load_egemaps_pooled(
    root: Path,
    split: str,
    anon_school: str,
    anon_class: str,
    anon_pid: str,
    session: str,
) -> np.ndarray | None:
```

**egemaps 特殊性**：
- egemaps (Extended Geneva Minimalistic Acoustic Parameter Set) 是一组音频统计特征
- 它是**池化特征**（pooled），即对整个音频会话计算统计值（均值、标准差等）
- 输出是固定维度向量（通常88维），而非时序序列

**文件格式支持**：
1. **Parquet 格式**（优先）：
   ```python
   parquet_path = base / "pooled.parquet"
   if parquet_path.exists():
       df = pd.read_parquet(parquet_path)
       return df.iloc[0].values.astype(np.float32)
   ```

2. **JSON 格式**（备选）：
   ```python
   json_path = base / "pooled.json"
   if json_path.exists():
       with open(json_path) as f:
           meta = json.load(f)
       if "features" in meta:
           return np.array(list(meta["features"].values()), dtype=np.float32)
   ```

**返回 None 的情况**：
- 文件不存在
- 文件格式不正确
- 加载失败

### discover_feature_sets() - 发现可用特征集

```python
def discover_feature_sets(
    root: Path,
    split: str,
    modality: str,
    limit: int = 5  # 扫描的最大参与者数
) -> dict[str, list[str]]:
```

**用途**：
- 探索数据目录结构
- 发现有哪些特征集可用
- 确定哪些特征集需要模型标签（如 ssl_embed）

**返回结构**：
```python
{
    "mel_mfcc": [],                     # 无子目录，不需要模型标签
    "vad": [],                          # 无子目录
    "ssl_embed": ["chinese-hubert-base", "wav2vec2-base"],  # 有模型标签
    "headpose_geom": [],                # 视频特征
    "vision_ssl_embed": ["dinov2-base", "clip-base"],
}
```

### list_file_ids() - 列出所有文件ID

```python
def list_file_ids(
    root: Path,
    split: str,
    limit: int = 0  # 0表示不限制
) -> list[tuple[str, str, str]]:
```

**用途**：
- 获取所有参与者的 (school, class, pid) 组合
- 用于生成提交文件或数据统计

**返回格式**：
```python
[
    ("school001", "class001", "pid001"),
    ("school001", "class001", "pid002"),
    ...
]
```

## 特征文件结构详解

### .npz 文件格式

`.npz` 是 NumPy 的压缩数组格式，可以存储多个数组：

```
sequence.npz 内容示例:
├── features: (T, D)        # 主特征数组
├── timestamps_ms: (T)      # 时间戳数组
└── valid_mask: (T)         # 有效帧掩码 (可选)
```

对于 `mel_mfcc` 特征：
```
mel_mfcc/sequence.npz:
├── mel_features: (T, 40)   # Mel频谱特征
├── mfcc_features: (T, 20)  # MFCC特征
├── timestamps_ms: (T)
└── valid_mask: (T)
```

### 特征维度参考

| 特征集 | 模态 | 类型 | 维度 | 说明 |
|--------|------|------|------|------|
| mel_mfcc | audio | 序列 | 60 | Mel(40) + MFCC(20) |
| vad | audio | 序列 | 1 | 语音活动检测 |
| ssl_embed | audio | 序列 | 768 | 自监督学习嵌入 |
| egemaps | audio | 池化 | 88 | 音频统计特征 |
| headpose_geom | video | 序列 | 6+ | 头部姿态(位置+旋转) |
| face_behavior | video | 序列 | 多维 | 面部行为特征 |
| body_pose | video | 序列 | 多维 | 身体姿态关键点 |
| vision_ssl_embed | video | 序列 | 768+ | 视觉SSL嵌入 |

## 错误处理机制

### FileNotFoundError

当特征文件不存在时抛出：
```python
if not seq_path.exists():
    raise FileNotFoundError(f"Missing sequence file: {seq_path}")
```

在数据集层面，这个错误会被捕获并跳过：
```python
try:
    seq = load_sequence(...)
except FileNotFoundError:
    pass  # 跳过不存在的特征
```

### KeyError

当文件格式不符合预期时抛出：
```python
if "timestamps_ms" not in data:
    raise KeyError(f"Missing 'timestamps_ms' in {seq_path}")
```

### ValueError

当数据形状不一致时抛出：
```python
if features.shape[0] != T:
    raise ValueError(f"Shape mismatch in {seq_path}")
```

## 使用示例

### 基本用法

```python
from pathlib import Path
from common.data.feature_io import load_sequence, load_egemaps_pooled

# 加载音频序列特征
seq = load_sequence(
    root=Path("/data/features"),
    split="train",
    anon_school="s001",
    anon_class="c001",
    anon_pid="p001",
    modality="audio",
    feature_set="mel_mfcc",
    session="A01"
)
print(f"Features shape: {seq.features.shape}")
print(f"Timestamps range: {seq.timestamps_ms[0]} - {seq.timestamps_ms[-1]} ms")
print(f"Valid frames: {seq.valid_mask.sum()} / {len(seq.valid_mask)}")

# 加载SSL嵌入（需要模型标签）
ssl_seq = load_sequence(
    root=Path("/data/features"),
    split="train",
    anon_school="s001",
    anon_class="c001",
    anon_pid="p001",
    modality="audio",
    feature_set="ssl_embed",
    session="A01",
    model_tag="chinese-hubert-base"
)

# 加载池化特征
egemaps = load_egemaps_pooled(
    root=Path("/data/features"),
    split="train",
    anon_school="s001",
    anon_class="c001",
    anon_pid="p001",
    session="A01"
)
print(f"egemaps shape: {egemaps.shape}")
```

### 探索可用特征

```python
from common.data.feature_io import discover_feature_sets

# 发现音频特征集
audio_features = discover_feature_sets(
    root=Path("/data/features"),
    split="train",
    modality="audio"
)
print("Available audio features:", audio_features)

# 发现视频特征集
video_features = discover_feature_sets(
    root=Path("/data/features"),
    split="train",
    modality="video"
)
print("Available video features:", video_features)
```

## 设计考量

### 为什么使用 .npz 格式？

1. **压缩效率**：NumPy 的压缩格式比纯文本更高效
2. **多维数组支持**：可以存储多个数组，适合时序数据
3. **跨平台兼容**：NumPy 是 Python 科学计算的标准库
4. **加载速度**：`np.load()` 比 JSON 解析更快

### 为什么区分序列特征和池化特征？

1. **时序建模需求**：序列特征可以用于 TCN 等时序模型
2. **计算效率**：池化特征已经压缩，减少计算负担
3. **信息互补**：池化特征提供全局统计，序列特征提供局部模式
4. **egemaps 特殊性**：它是音频分析的标准化特征集，已经设计为池化输出

### 时间戳的作用

时间戳是数据对齐的关键：
- 不同特征可能有不同的采样率
- 通过时间戳可以对齐到统一网格
- 用于计算帧间距离，确定最近邻插值