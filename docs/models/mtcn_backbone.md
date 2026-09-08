# mtcn_backbone.py - 多模态TCN骨干网络详解

## 文件概述

`mtcn_backbone.py` 定义了项目的核心神经网络架构 —— MTCN (Multimodal Temporal Convolutional Network)。这是一个专门为多模态音视频特征设计的时序建模骨干网络。

## 核心组件

### 1. BackboneConfig 配置类

```python
@dataclass
class BackboneConfig:
    # 输入特征维度字典
    audio_group_dims: dict[str, int] = field(default_factory=dict)       # 音频序列特征维度
    audio_pooled_group_dims: dict[str, int] = field(default_factory=dict) # 音频池化特征维度
    video_group_dims: dict[str, int] = field(default_factory=dict)       # 视频特征维度

    # 网络结构参数
    d_adapter: int = 64      # 适配层输出维度
    d_model: int = 256       # TCN建模维度
    tcn_layers: int = 6      # TCN层数
    tcn_kernel_size: int = 3 # TCN卷积核大小
    asp_alpha: float = 0.5   # ASP中VAD信号权重
    asp_beta: float = 0.5    # ASP中QC信号权重
    dropout: float = 0.1     # Dropout比率

    # 会话嵌入
    n_sessions: int = 4      # 会话类型数
    d_session: int = 16      # 会话嵌入维度
    d_shared: int = 256      # 最终输出维度
```

**为什么需要这些配置？**

- `audio_group_dims` / `video_group_dims`: 不同特征有不同维度，需要分别适配
- `d_adapter`: 统一维度便于后续融合
- `d_model`: TCN的处理维度，影响模型容量
- `tcn_layers` / `tcn_kernel_size`: 控制感受野大小和时序建模能力

**感受野计算**：
```
感受野 = 1 + (kernel_size - 1) × Σ(dilation^i) for i in [0, layers-1]

例: kernel=3, layers=6
膨胀率: [1, 2, 4, 8, 16, 32]
感受野 = 1 + 2 × (1+2+4+8+16+32) = 1 + 2×63 = 127
```

### 2. GroupAdapter - 特征适配层

```python
class GroupAdapter(nn.Module):
    def __init__(self, d_in: int, d_out: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_in)    # 输入归一化
        self.proj = nn.Linear(d_in, d_out) # 投影层
        self.drop = nn.Dropout(dropout)    # Dropout
```

**前向传播**：
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    # x: (B, T, d_in) 或 (N, T, d_in)
    # 归一化 → 投影 → 激活 → Dropout
    return self.drop(F.gelu(self.proj(self.norm(x))))
```

**设计原理**：

1. **LayerNorm**: 在时序维度上归一化，稳定训练
2. **GELU激活**: 平滑的非线性，比ReLU更稳定
3. **Dropout**: 防止过拟合
4. **输出维度**: 统一为 d_adapter，便于后续融合

**为什么需要适配层？**

不同特征有不同的原始维度：
- mel_mfcc: 60维
- ssl_embed: 768维
- headpose_geom: 6维
- ...

直接拼接会导致某些特征被淹没。适配层将不同维度投影到统一维度，实现公平融合。

### 3. ModalityFusion - 模态融合层

```python
class ModalityFusion(nn.Module):
    def __init__(self, n_groups: int, d_adapter: int, d_model: int) -> None:
        super().__init__()
        # 将所有特征组的适配特征拼接后投影到 d_model
        self.proj = nn.Linear(n_groups * d_adapter, d_model)

    def forward(self, groups: list[torch.Tensor]) -> torch.Tensor:
        return self.proj(torch.cat(groups, dim=-1))
```

**输入输出示例**：
```python
# 假设有3个音频特征组
# mel_mfcc: (B, T, 64)
# ssl_embed: (B, T, 64)
# vad: (B, T, 64)

groups = [mel_adapted, ssl_adapted, vad_adapted]  # 每个 (B, T, 64)
concat = torch.cat(groups, dim=-1)  # (B, T, 192)
output = self.proj(concat)  # (B, T, 256) ← d_model
```

**为什么是早期融合？**

- 在模态内部先融合多个特征组
- 减少后续TCN的输入维度
- 让不同特征组在时序建模前充分交互

### 4. DilatedResidualBlock - 膨胀残差块

这是TCN的核心构建块：

```python
class DilatedResidualBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        kernel_size: int,
        dilation: int,         # 膨胀率
        dropout: float
    ) -> None:
        super().__init__()
        # 计算填充以保持序列长度
        padding = (kernel_size - 1) * dilation // 2
        
        # 权重归一化卷积层
        self.conv1 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(d_model, d_model, kernel_size, dilation=dilation, padding=padding)
        )
        self.conv2 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(d_model, d_model, kernel_size, dilation=dilation, padding=padding)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
```

**前向传播详解**：

```python
def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    residual = x  # 残差连接
    T = x.size(1)
    
    # Block 1
    h = self.norm1(x)        # 层归一化
    h = h.transpose(1, 2)    # (B, T, D) → (B, D, T) 用于Conv1d
    h = self.conv1(h)        # 膨胀卷积
    h = h[:, :, :T]          # 裁剪到原始长度
    h = F.gelu(h)            # 激活
    h = self.drop(h)         # Dropout
    
    # Block 2
    h = h.transpose(1, 2)    # (B, D, T) → (B, T, D)
    h = self.norm2(h)
    h = h.transpose(1, 2)
    h = self.conv2(h)[:, :, :T]  # 另一个膨胀卷积
    h = self.drop(h)
    h = h.transpose(1, 2)    # 回到 (B, T, D)
    
    # 残差连接
    out = h + residual
    out = out * mask.unsqueeze(-1).float()  # 应用掩码
    return out
```

**膨胀卷积原理**：

```
普通卷积 (dilation=1):
    输入: [a, b, c, d, e, f, g, h]
    核:   [w1, w2, w3]
    输出: w1*a + w2*b + w3*c, w1*b + w2*c + w3*d, ...

膨胀卷积 (dilation=2):
    输入: [a, b, c, d, e, f, g, h]
    核:   [w1, _, w2, _, w3]  (_表示跳跃)
    输出: w1*a + w2*c + w3*e, w1*b + w2*d + w3*f, ...

膨胀卷积 (dilation=4):
    输入: [a, b, c, d, e, f, g, h]
    核:   [w1, _, _, _, w2, _, _, _, w3]
    输出: w1*a + w2*e + w3*i, ...

感受野随膨胀率指数增长!
```

**残差连接的作用**：

1. **梯度流通**: 允许梯度直接反向传播，缓解梯度消失
2. **信息保留**: 保留低层特征，让网络学习残差映射
3. **训练稳定性**: 即使某些层输出为0，也能保证前向传播

**权重归一化 (Weight Normalization)**：

```python
nn.utils.parametrizations.weight_norm(nn.Conv1d(...))
```

比Batch Normalization更适合序列数据：
- 不依赖于批次统计量
- 训练更快
- 序列长度变化时更稳定

### 5. TCN - 时序卷积网络

```python
class TCN(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layers: int,
        kernel_size: int,
        dropout: float
    ) -> None:
        super().__init__()
        # 每层的膨胀率呈指数增长: 1, 2, 4, 8, ..., 2^(n-1)
        self.blocks = nn.ModuleList([
            DilatedResidualBlock(d_model, kernel_size, dilation=2**i, dropout=dropout)
            for i in range(n_layers)
        ])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, mask)
        return x
```

**感受野可视化**：

```
层数:        0      1       2        3        4         5
膨胀率:      1      2       4        8        16        32
感受野:      3      7       15       31       63        127

输出点可以看到的时间范围:
Layer 0:  [xxx]                                (3帧)
Layer 1:  [xxxxxxx]                            (7帧)
Layer 2:  [xxxxxxxxxxxxxxx]                    (15帧)
...
Layer 5:  [xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx]    (127帧)

如果每帧40ms:
127帧 = 127 × 40ms = 5.08秒
模型可以"看到"过去5秒的信息!
```

**TCN vs RNN vs Transformer**：

| 特性 | TCN | RNN/LSTM | Transformer |
|------|-----|----------|-------------|
| 并行性 | ✅ 完全并行 | ❌ 顺序处理 | ✅ 完全并行 |
| 感受野 | ✅ 可控 | ❌ 必须完整序列 | ✅ 全局但二次复杂度 |
| 内存 | ✅ 线性 | ⚠️ 需存储隐藏状态 | ❌ 二次增长 |
| 梯度流 | ✅ 稳定 | ❌ 容易梯度消失 | ✅ 稳定 |
| 速度 | ✅ 快 | ⚠️ 慢 | ⚠️ 中等 |

**为什么选择TCN？**

- 音视频特征具有局部时序相关性
- TCN的卷积结构很适合这种局部模式
- 训练速度快，适合大规模数据
- 内存占用稳定

### 6. ASP - 注意力统计池化

```python
class ASP(nn.Module):
    """Attentive Statistics Pooling with VAD and quality control signals."""

    def __init__(self, d_model: int, alpha: float = 0.5, beta: float = 0.5) -> None:
        super().__init__()
        self.attn = nn.Linear(d_model, 1)  # 注意力投影
        self.alpha = nn.Parameter(torch.tensor(alpha))  # VAD权重（可学习）
        self.beta = nn.Parameter(torch.tensor(beta))    # QC权重（可学习）
```

**前向传播详解**：

```python
def forward(
    self,
    x: torch.Tensor,    # (B, T, D) - TCN输出
    mask: torch.Tensor, # (B, T) - 有效性掩码
    vad: torch.Tensor,  # (B, T) - VAD信号
    qc: torch.Tensor,   # (B, T) - 质量控制信号
) -> torch.Tensor:
    """
    返回: (B, 2*D) - 拼接的均值和标准差
    """
    # 1. 计算注意力分数
    e = self.attn(x).squeeze(-1)  # (B, T) ← 从特征投影得到
    
    # 2. 结合外部信号
    e = e + self.alpha * vad + self.beta * qc
    
    # 3. 应用掩码（无效位置设为负无穷）
    e = e.masked_fill(~mask, float("-inf"))
    
    # 4. Softmax归一化得到注意力权重
    w = F.softmax(e, dim=-1)  # (B, T), 和为1
    w = w.masked_fill(~mask, 0.0)  # 避免NaN
    
    # 5. 计算加权均值
    w_unsq = w.unsqueeze(-1)  # (B, T, 1)
    mean = (w_unsq * x).sum(dim=1)  # (B, D)
    
    # 6. 计算加权标准差
    diff = x - mean.unsqueeze(1)  # (B, T, D)
    var = (w_unsq * diff ** 2).sum(dim=1)
    std = torch.sqrt(var.clamp(min=1e-8))  # 避免sqrt(0)
    
    # 7. 拼接均值和标准差
    return torch.cat([mean, std], dim=-1)  # (B, 2*D)
```

**注意力机制图解**：

```
输入序列: x = [x1, x2, x3, x4, x5, x6, x7]  (假设7帧)
掩码:     mask = [1, 1, 1, 0, 1, 1, 0]     (第4帧和第7帧无效)
VAD:      vad =  [0.9, 0.3, 0.8, 0.0, 0.1, 0.95, 0.0]
QC:       qc =   [0.8, 0.5, 0.9, 0.0, 0.2, 0.85, 0.0]

计算过程:
1. 基础注意力分数 (attn(x)):
   e = [0.1, 0.2, 0.15, 0.1, 0.05, 0.25, 0.05]

2. 结合VAD和QC (假设 alpha=0.5, beta=0.5):
   e = e + 0.5*vad + 0.5*qc
     = [0.95, 0.55, 1.0, 0.0, 0.20, 1.175, 0.025]

3. 应用掩码:
   e = [0.95, 0.55, 1.0, -inf, 0.20, 1.175, -inf]

4. Softmax:
   w = [0.28, 0.17, 0.30, 0, 0.10, 0.35, 0]
   
   注意: 高VAD/QC的帧(1,3,6)获得了更高权重

5. 加权统计:
   mean = Σ(w_i * x_i)  (高权重帧贡献更大)
   std = sqrt(Σ(w_i * (x_i - mean)²))
```

**为什么需要ASP而非简单平均？**

1. **选择性关注**: 自动学习哪些时间帧更重要
2. **结合VAD**: 强制关注语音活动帧，避免静音
3. **质量控制**: 优先使用高质量的帧（面部清晰可见等）
4. **统计特征**: 均值+标准差比单一平均更丰富

**VAD和QC信号的作用**：

- **VAD (Voice Activity Detection)**: 标记哪些时间片段有语音活动
  - 高VAD → 更可能包含说话内容 → 心理相关
  - 低VAD → 静音或噪音 → 不太相关

- **QC (Quality Control)**: 视频质量信号
  - 高QC → 面部清晰可见 → 特征可靠
  - 低QC → 面部遮挡/模糊 → 特征不可靠

### 7. MTCNBackbone - 完整骨干网络

```python
class MTCNBackbone(nn.Module):
    def __init__(self, cfg: BackboneConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # 1. 创建特征适配器
        self.audio_adapters = nn.ModuleDict({
            name: GroupAdapter(d_in, cfg.d_adapter, cfg.dropout)
            for name, d_in in cfg.audio_group_dims.items()
        })
        self.video_adapters = nn.ModuleDict({
            name: GroupAdapter(d_in, cfg.d_adapter, cfg.dropout)
            for name, d_in in cfg.video_group_dims.items()
        })
        self.audio_pooled_adapters = nn.ModuleDict({
            name: nn.Sequential(...)  # 池化特征不需要时序处理
            for name, d_in in cfg.audio_pooled_group_dims.items()
        })

        # 2. 模态融合
        self.audio_fusion = ModalityFusion(
            len(self.audio_group_names), cfg.d_adapter, cfg.d_model
        )
        self.video_fusion = ModalityFusion(
            len(self.video_group_names), cfg.d_adapter, cfg.d_model
        )

        # 3. TCN时序建模
        self.audio_tcn = TCN(cfg.d_model, cfg.tcn_layers, cfg.tcn_kernel_size, cfg.dropout)
        self.video_tcn = TCN(cfg.d_model, cfg.tcn_layers, cfg.tcn_kernel_size, cfg.dropout)

        # 4. ASP池化
        self.audio_asp = ASP(cfg.d_model, cfg.asp_alpha, cfg.asp_beta)
        self.video_asp = ASP(cfg.d_model, cfg.asp_alpha, cfg.asp_beta)

        # 5. 最终融合MLP
        fusion_in = 2 * cfg.d_model * 2  # 音频ASP + 视频ASP
        fusion_in += len(self.audio_pooled_group_names) * cfg.d_adapter
        fusion_in += cfg.d_session
        
        self.session_embed = nn.Embedding(cfg.n_sessions, cfg.d_session)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_in, cfg.d_shared),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_shared, cfg.d_shared),
        )
```

**前向传播完整流程**：

```python
def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    # batch 包含:
    # - audio_groups: dict[str, Tensor] (N, T, D)
    # - video_groups: dict[str, Tensor] (N, T, D)
    # - audio_pooled_groups: dict[str, Tensor] (N, D)
    # - mask_audio, mask_video: (N, T)
    # - vad_signal, qc_quality: (N, T)
    # - session_idx: (N,)

    # 1. 特征适配
    audio_adapted = [
        self.audio_adapters[n](batch["audio_groups"][n])
        for n in self.audio_group_names
    ]
    video_adapted = [
        self.video_adapters[n](batch["video_groups"][n])
        for n in self.video_group_names
    ]

    # 2. 模态内融合
    a = self.audio_fusion(audio_adapted)  # (N, T, d_model)
    v = self.video_fusion(video_adapted)  # (N, T, d_model)

    # 3. 应用模态掩码
    a = a * batch["mask_audio"].unsqueeze(-1).float()
    v = v * batch["mask_video"].unsqueeze(-1).float()

    # 4. TCN时序建模
    a = self.audio_tcn(a, batch["mask_audio"])  # (N, T, d_model)
    v = self.video_tcn(v, batch["mask_video"])  # (N, T, d_model)

    # 5. ASP池化
    z_a = self.audio_asp(a, batch["mask_audio"], batch["vad_signal"], batch["qc_quality"])
    z_v = self.video_asp(v, batch["mask_video"], batch["vad_signal"], batch["qc_quality"])

    # 6. 融合所有特征
    parts = [z_a, z_v]  # (N, 2*d_model) each
    parts.extend(
        self.audio_pooled_adapters[name](batch["audio_pooled_groups"][name])
        for name in self.audio_pooled_group_names
    )
    parts.append(self.session_embed(batch["session_idx"]))  # (N, d_session)

    z = torch.cat(parts, dim=-1)
    return self.fusion_mlp(z)  # (N, d_shared)
```

## 完整数据流图解

```
输入数据 (每个会话一个样本):
┌──────────────────────────────────────────────────────────────────────────┐
│  audio_groups:                                                           │
│    mel_mfcc: (N, T, 60)                                                 │
│    vad: (N, T, 1)                                                       │
│    ssl_embed: (N, T, 768)                                               │
│  video_groups:                                                           │
│    headpose_geom: (N, T, 6)                                             │
│    face_behavior: (N, T, D)                                             │
│    body_pose: (N, T, D)                                                 │
│    vision_ssl_embed: (N, T, 768)                                        │
│  audio_pooled_groups:                                                    │
│    egemaps: (N, 88)                                                     │
│  mask_audio, mask_video: (N, T)                                         │
│  vad_signal, qc_quality: (N, T)                                         │
│  session_idx: (N,)                                                      │
└──────────────────────────────────────────────────────────────────────────┘
                                    ↓

1. GroupAdapter (特征适配)
┌──────────────────────────────────────────────────────────────────────────┐
│  mel_mfcc: (N, T, 60) ──[Adapter]──► (N, T, 64)                        │
│  ssl_embed: (N, T, 768) ──[Adapter]──► (N, T, 64)                      │
│  vad: (N, T, 1) ──[Adapter]──► (N, T, 64)                              │
│  headpose_geom: (N, T, 6) ──[Adapter]──► (N, T, 64)                    │
│  ...                                                                     │
└──────────────────────────────────────────────────────────────────────────┘
                                    ↓

2. ModalityFusion (模态融合)
┌──────────────────────────────────────────────────────────────────────────┐
│  Audio: concat(3 × (N, T, 64)) = (N, T, 192) ──[Proj]──► (N, T, 256)   │
│  Video: concat(4 × (N, T, 64)) = (N, T, 256) ──[Proj]──► (N, T, 256)   │
└──────────────────────────────────────────────────────────────────────────┘
                                    ↓

3. TCN (时序建模)
┌──────────────────────────────────────────────────────────────────────────┐
│  Audio: (N, T, 256) ──[TCN×6]──► (N, T, 256)                            │
│  Video: (N, T, 256) ──[TCN×6]──► (N, T, 256)                            │
│                                                                          │
│  感受野: 127帧 ≈ 5秒                                                     │
└──────────────────────────────────────────────────────────────────────────┘
                                    ↓

4. ASP (池化)
┌──────────────────────────────────────────────────────────────────────────┐
│  Audio: (N, T, 256) ──[ASP]──► (N, 512) [mean+std]                     │
│  Video: (N, T, 256) ──[ASP]──► (N, 512) [mean+std]                     │
└──────────────────────────────────────────────────────────────────────────┘
                                    ↓

5. 最终融合
┌──────────────────────────────────────────────────────────────────────────┐
│  Input:                                                                  │
│    Audio ASP: (N, 512)                                                  │
│    Video ASP: (N, 512)                                                  │
│    egemaps: (N, 64)                                                     │
│    Session embed: (N, 16)                                               │
│  Total: 512 + 512 + 64 + 16 = 1104                                      │
│                                                                          │
│  Output: (N, 1104) ──[MLP]──► (N, d_shared)                            │
└──────────────────────────────────────────────────────────────────────────┘
```

## 初始化策略

```python
def _init_weights(self) -> None:
    for m in self.modules():
        if isinstance(m, nn.Linear):
            # Xavier初始化，适合GELU激活
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            # 嵌入层标准初始化
            nn.init.normal_(m.weight, std=0.02)
```

**为什么选择Xavier初始化？**

- 保持各层输入输出的方差一致
- 特别适合带有GELU等激活函数的网络
- 训练更稳定，收敛更快

## 使用示例

```python
from common.models.mtcn_backbone import BackboneConfig, MTCNBackbone

# 配置
bb_cfg = BackboneConfig(
    audio_group_dims={"mel_mfcc": 60, "vad": 1, "ssl_embed": 768},
    audio_pooled_group_dims={"egemaps": 88},
    video_group_dims={"headpose_geom": 6, "face_behavior": 128},
    d_adapter=64,
    d_model=256,
    tcn_layers=6,
    tcn_kernel_size=3,
    dropout=0.2,
    d_shared=256,
)

# 创建模型
backbone = MTCNBackbone(bb_cfg)

# 前向传播
output = backbone(flat_batch)  # (N, 256)
```