"""多模态时序卷积网络骨干 (MTCNBackbone)

========================================================================
模型原理
========================================================================

本模块解决的核心问题：
    从多模态（音频+视频）的变长时序数据中，提取固定维度的会话级表示，
    用于情感识别或行为预测的下游任务。

整体处理流程（分而治之）：
    原始多模态特征
        → [Step 1] GroupAdapter：特征对齐（各维度 → 统一 d_adapter）
        → [Step 2] ModalityFusion：模态内融合（多组 → 一组 d_model）
        → [Step 3] TCN：时序建模（捕获多尺度时序模式）
        → [Step 4] ASP：统计池化（变长序列 → 固定向量）
        → [Step 5] 跨模态融合 + MLP：最终会话级表示

========================================================================
涉及的核心技术
========================================================================

1. 膨胀卷积（Dilated Convolution）
   - 原理：在卷积核元素间插入空洞，扩大感受野而不增加参数量
   - 普通卷积（dilation=1）：感受野 = kernel_size
   - 膨胀卷积（dilation=d）：感受野 = kernel_size + (kernel_size-1)×(d-1)
   - 例：kernel=3, dilation=4 → 感受野 = 3 + 2×3 = 9，但参数量仍是3
   - 多层指数膨胀（1,2,4,8,...）：感受野指数增长，高效覆盖长时序

2. 残差连接（Residual Connection）
   - 原理：output = F(x) + x，网络只需学习残差 F(x)
   - 作用：为梯度提供"高速公路"，防止深层网络梯度消失
   - 来自 ResNet（He et al., 2016）

3. 权重归一化（Weight Normalization）
   - 原理：W = g × (v / ||v||)，将权重分解为方向和幅度
   - 作用：加速收敛，稳定梯度，比 BatchNorm 更适合序列任务

4. 注意力统计池化（Attentive Statistics Pooling, ASP）
   - 原理：加权求均值和标准差，权重由注意力+VAD+QC决定
   - 优于均值池化：标准差捕获分布形状（如情绪激动程度）
   - VAD（语音活动检测）：说话时刻权重高，静音权重低
   - QC（质量控制）：高质量帧权重高，模糊/遮挡帧权重低

5. 时序卷积网络（TCN）vs RNN vs Transformer
   - TCN 优势：并行计算、梯度稳定、感受野可控
   - 本任务选 TCN 的原因：中等规模数据集（Transformer 易过拟合），
     序列长度适中（TCN 感受野足够），需要快速训练

该模块包含:
    1. 特征适配层 (GroupAdapter): 将不同维度的特征投影到统一维度
    2. 模态融合层 (ModalityFusion): 在模态内部融合多个特征组
    3. 膨胀残差卷积块 (DilatedResidualBlock): TCN的基本单元
    4. 时序卷积网络 (TCN): 多尺度时序建模
    5. 注意力统计池化 (ASP): 结合VAD和QC信号的时序聚合

"""

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class BackboneConfig:
    """骨干网络配置类

    属性:
        audio_group_dims: 音频时序特征维度字典 {特征名: 维度}
            例: {"mfcc": 40, "wav2vec": 768}
            不同特征提取器产生不同维度，需要各自独立适配

        audio_pooled_group_dims: 音频池化特征维度字典
            这类特征已经在时间维度上做了全局池化（无时序维度），
            不经过 TCN，直接在最终融合阶段拼接

        video_group_dims: 视频特征维度字典
            例: {"efficientnet": 1280, "openface_au": 35}

        d_adapter: 适配器输出维度 (默认64)
            所有特征组被投影到这个维度，实现维度对齐

        d_model: TCN 输入/输出维度 (默认256)
            模态融合后的统一特征维度，也是 TCN 的工作维度

        tcn_layers: TCN 层数 (默认4)
            决定感受野大小：
            4层,kernel=3 → RF = 1 + 2×(1+2+4+8) = 31 帧
            6层,kernel=3 → RF = 1 + 2×(1+2+4+8+16+32) = 127 帧

        tcn_kernel_size: 卷积核大小 (默认3)
            每个膨胀卷积核的实际大小

        asp_alpha: VAD 信号权重参数 (默认0.5)
            初始值，之后由反向传播自动调整

        asp_beta: QC 信号权重参数 (默认0.5)
            初始值，之后由反向传播自动调整

        dropout: Dropout 比率 (默认0.1)
            在适配器、TCN等模块中使用，防止过拟合

        n_sessions: 会话数量 (默认4)
            用于会话嵌入（Session Embedding），捕获不同采集场景的系统性差异

        d_session: 会话嵌入维度 (默认16)
            较小的维度，仅作为辅助信息

        d_shared: 最终输出维度 (默认256)
            骨干网络输出的会话级表示维度，供下游任务头使用
    """
    audio_group_dims: dict[str, int] = field(default_factory=dict)
    audio_pooled_group_dims: dict[str, int] = field(default_factory=dict)
    video_group_dims: dict[str, int] = field(default_factory=dict)

    d_adapter: int = 64
    d_model: int = 256
    tcn_layers: int = 4
    tcn_kernel_size: int = 3
    asp_alpha: float = 0.5
    asp_beta: float = 0.5
    dropout: float = 0.1

    n_sessions: int = 4
    d_session: int = 16
    d_shared: int = 256

    use_cross_modal: bool = False
    cm_n_heads: int = 1

class GroupAdapter(nn.Module):
    """特征适配层

    将不同维度的输入特征投影到统一的 d_adapter 维度。

    问题背景：
        不同特征提取器产生不同维度：
            MFCC → 40维，Mel Spectrogram → 128维，Wav2Vec → 768维
        无法直接拼接或融合，需要先对齐到统一维度。

    结构：LayerNorm → Linear → GELU → Dropout

    各层作用：
        LayerNorm：对输入做层归一化，稳定特征分布，适合时序数据
                   （BatchNorm 依赖 batch 统计量，时序场景不稳定）
        Linear：   线性投影，实现维度变换（升维或降维都可）
        GELU：     非线性激活，比 ReLU 更平滑（无硬截断），来自 GPT 系列
        Dropout：  随机置零，防止过拟合，只在训练时生效

    参数:
        d_in:    输入特征维度（如 768）
        d_out:   输出维度，通常是 d_adapter（如 64）
        dropout: Dropout 比率
    """
    def __init__(self, d_in: int, d_out: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_in)   # LayerNorm 对最后一维归一化，适合序列数据
        self.proj = nn.Linear(d_in, d_out)  # 线性投影到统一维度
        self.drop = nn.Dropout(dropout)  # 训练时随机置零，增强泛化

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 链式调用：LayerNorm → Linear → GELU → Dropout
        # GELU(x) = x × Φ(x)，Φ 为标准正态分布 CDF，比 ReLU 更平滑
        return self.drop(F.gelu(self.proj(self.norm(x))))

class ModalityFusion(nn.Module):
    """模态内融合层

    将同一模态的多个特征组（已对齐到 d_adapter 维度）融合为单一表示。

    问题背景：
        一个模态可能有多组特征，例如音频模态包含：
            - MFCC（声学特征）→ 适配后 d_adapter 维
            - Wav2Vec（自监督特征）→ 适配后 d_adapter 维
            - Mel Spectrogram → 适配后 d_adapter 维
        需要融合为一个统一向量，输入 TCN 进行时序建模。

    方法：拼接（Concatenation）后线性投影
        拼接：[feat1; feat2; feat3] → (n_groups × d_adapter) 维
        投影：→ d_model 维

    为什么用拼接而非逐元素加法？
        拼接保留各特征组的独立信息，让线性层自主学习融合权重；
        逐元素加法要求各组特征意义完全对应，限制太强。

    参数:
        n_groups:  特征组数量
        d_adapter: 每组特征的维度
        d_model:   融合后的输出维度
    """
    def __init__(self, n_groups: int, d_adapter: int, d_model: int) -> None:
        super().__init__()
        # 输入：n_groups 组特征拼接后的总维度
        self.proj = nn.Linear(n_groups * d_adapter, d_model)

    def forward(self, groups: list[torch.Tensor]) -> torch.Tensor:
        # groups: 每个元素 shape (B, T, d_adapter)
        # cat 后: (B, T, n_groups × d_adapter) → 线性投影 → (B, T, d_model)
        return self.proj(torch.cat(groups, dim=-1))


class DilatedResidualBlock(nn.Module):
    """膨胀残差卷积块

    TCN 的基本构建单元，使用膨胀卷积捕获多尺度时序模式。

    ====================================================================
    膨胀卷积（Dilated Convolution）原理
    ====================================================================

    普通卷积（dilation=1）：
        输入: [x1, x2, x3, x4, x5]
        kernel: [w1, w2, w3]（连续采样）
        感受野 = kernel_size = 3

    膨胀卷积（dilation=2）：
        输入: [x1, x2, x3, x4, x5]
        kernel: [w1,  _, w2,  _, w3]（跳跃采样，_表示跳过）
        感受野 = kernel_size + (kernel_size-1)×(dilation-1) = 3+2×1 = 5
        参数量与普通卷积相同，但感受野翻倍！

    多层堆叠（dilation=1,2,4,8）：
        层0: 感受野=3
        层1: 感受野=3+2×2=7（在层0基础上扩展）
        层2: 感受野=7+2×4=15
        层3: 感受野=15+2×8=31

    ====================================================================
    结构（双卷积块，类似 ResNet BottleNeck）
    ====================================================================

        输入 x
         │
         ├──────────────── residual（跳过连接）
         │
         ▼
        LayerNorm1         # 归一化稳定训练
         ▼
        transpose(1,2)     # (B,T,D) → (B,D,T)，Conv1d 要求通道在第2维
         ▼
        Conv1d(dilation)   # 膨胀卷积，提取时序模式
         ▼
        [:T]               # 截断到原始长度（padding 可能使输出稍长）
         ▼
        GELU               # 非线性激活
         ▼
        Dropout
         ▼
        transpose(1,2)     # (B,D,T) → (B,T,D)
        LayerNorm2
        transpose(1,2)     # 再转回 Conv1d 格式
         ▼
        Conv1d(dilation)   # 第二次卷积，特征变换
         ▼
        [:T] → Dropout
         ▼
        transpose(1,2)     # (B,D,T) → (B,T,D)
         ▼
        h + residual       # 残差连接：防止梯度消失，让网络只学差量
         ▼
        × mask             # 掩码：无效时间步（padding）置零
         ▼
        输出

    参数:
        d_model:    特征维度（输入=输出，残差连接要求相同维度）
        kernel_size: 卷积核大小
        dilation:   膨胀率（TCN 中指数增长：1,2,4,8,...）
        dropout:    Dropout 比率
    """
    def __init__(
        self, d_model: int, kernel_size: int, dilation: int, dropout: float
    ) -> None:
        super().__init__()
        # padding 保证输出长度 = 输入长度（same padding）
        # 公式：padding = (kernel_size - 1) × dilation // 2
        # 例：kernel=3, dilation=4 → padding = 2×4//2 = 4
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(d_model, d_model, kernel_size, dilation=dilation, padding=padding)
        )
        # weight_norm：将权重分解为方向向量 v 和幅度标量 g
        #   W = g × (v / ||v||)
        # 作用：加速收敛，稳定梯度，比 BatchNorm 更适合 1D 时序卷积
        self.conv2 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(d_model, d_model, kernel_size, dilation=dilation, padding=padding)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x:    (B, T, D) — 时序特征，B=batch, T=时间步, D=特征维度
            mask: (B, T)    — bool 掩码，True 表示有效位置
        返回:
            out: (B, T, D) — 与输入同形状
        """
        residual = x          # 保存残差，用于后续加法
        T = x.size(1)         # 原始时间步数，用于截断

        # ---- 第一个子块 ----
        h = self.norm1(x)               # LayerNorm：对特征维度归一化
        h = h.transpose(1, 2)           # (B,T,D) → (B,D,T)，符合 Conv1d 输入格式
        h = self.conv1(h)[:, :, :T]     # 膨胀卷积 + 截断（same length）
        h = F.gelu(h)                   # 平滑非线性激活
        h = self.drop(h)                # Dropout

        # ---- 第二个子块 ----
        h = h.transpose(1, 2)           # (B,D,T) → (B,T,D)
        h = self.norm2(h)               # LayerNorm
        h = h.transpose(1, 2)           # (B,T,D) → (B,D,T)
        h = self.conv2(h)[:, :, :T]     # 第二次膨胀卷积 + 截断
        h = self.drop(h)
        h = h.transpose(1, 2)           # (B,D,T) → (B,T,D)

        out = h + residual              # 残差连接：梯度可直接通过，防止梯度消失
        out = out * mask.unsqueeze(-1).float()  # 掩码：padding 位置置零，不污染后续计算
        return out


class TCN(nn.Module):
    """时序卷积网络 (Temporal Convolutional Network)

    由多个膨胀残差块堆叠而成，膨胀率指数增长（1, 2, 4, 8, ...）。

    ====================================================================
    为什么选 TCN 而不是 RNN 或 Transformer？
    ====================================================================

    | 架构        | 优势                          | 劣势                        |
    |-------------|-------------------------------|----------------------------|
    | TCN（本模型）| 并行、梯度稳定、感受野可控    | 感受野有上限                |
    | RNN/LSTM    | 理论可建模任意长依赖          | 串行慢、梯度不稳定          |
    | Transformer | 全局注意力、效果最好          | O(T²) 复杂度、需要大量数据  |

    本任务选 TCN 的原因：
        - 中等规模数据集（~1000样本）→ Transformer 易过拟合
        - 序列长度适中（~100帧）→ TCN 感受野（31帧）已足够
        - 需要快速迭代训练 → TCN 并行计算优势明显

    ====================================================================
    感受野计算
    ====================================================================

    感受野（Receptive Field）= 一个输出单元能"看到"的输入范围

    公式：RF = 1 + (kernel_size - 1) × Σ(2^i)，i ∈ [0, n_layers-1]

    4 层（默认），kernel=3：
        RF = 1 + 2 × (1 + 2 + 4 + 8) = 1 + 2×15 = 31 帧

    6 层，kernel=3：
        RF = 1 + 2 × (1 + 2 + 4 + 8 + 16 + 32) = 127 帧

    参数:
        d_model:     特征维度
        n_layers:    TCN 层数（控制感受野大小）
        kernel_size: 卷积核大小
        dropout:     Dropout 比率
    """
    def __init__(
        self, d_model: int, n_layers: int, kernel_size: int, dropout: float
    ) -> None:
        super().__init__()
        # ModuleList 保证参数被正确注册到模型，可被 optimizer 追踪
        # 膨胀率指数增长：2^0=1, 2^1=2, 2^2=4, 2^3=8
        # 这样感受野在每一层都能翻倍扩展，覆盖更长的时序依赖
        self.blocks = nn.ModuleList([
            DilatedResidualBlock(d_model, kernel_size, dilation=2**i, dropout=dropout)
            for i in range(n_layers)
        ])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x:    (B, T, D) — 时序特征
            mask: (B, T)    — bool 掩码
        返回:
            x: (B, T, D) — 经过多层膨胀残差块后的时序特征
        """
        for block in self.blocks:
            x = block(x, mask)  # 每层感受野翻倍，信息逐层汇聚
        return x


class ASP(nn.Module):
    """注意力统计池化 (Attentive Statistics Pooling)

    将变长时序（B, T, D）压缩为固定大小的统计向量（B, 2D）。

    ====================================================================
    为什么需要统计池化？
    ====================================================================

    问题：不同样本的时序长度 T 不同，无法直接用于固定尺寸的分类器。
    方案：对时序维度做加权聚合，输出固定大小向量。

    为什么要同时输出均值和标准差？
        均值：捕获平均状态（如平均情感强度）
        标准差：捕获分布形状（如情感波动程度）
        例：两个人说话情感均值相同，但一个平稳（小std）、一个激动（大std）
        仅用均值无法区分，加入 std 提供更丰富的信息。

    ====================================================================
    为什么结合 VAD 和 QC 信号？
    ====================================================================

    VAD（Voice Activity Detection，语音活动检测）：
        vad≈1：此时刻有人说话 → 提高权重，关注有效语音
        vad≈0：静音或噪声时刻 → 降低权重，减少噪声干扰

    QC（Quality Control，质量控制信号）：
        qc≈1：帧质量高（清晰、正脸、无遮挡）→ 提高权重
        qc≈0：帧质量低（模糊、侧脸、遮挡）→ 降低权重

    这两个信号提供了领域先验知识，比纯数据驱动的注意力更可靠。

    ====================================================================
    计算流程
    ====================================================================

        x: (B, T, D)      — TCN 输出的时序特征
        mask: (B, T)      — bool 掩码
        vad: (B, T)       — 语音活动信号
        qc: (B, T)        — 质量控制信号

        Step 1: 计算注意力分数
            e = Linear(x) + α×vad + β×qc    shape: (B, T)
            α, β 是可学习标量（nn.Parameter），初始化为 0.5

        Step 2: 掩码 + Softmax 归一化
            e[~mask] = -inf                  无效位置填充负无穷
            w = softmax(e, dim=-1)           → (B, T)，Σw_t = 1
            w[~mask] = 0                     避免 NaN（全掩码时 softmax 为 NaN）

        Step 3: 加权统计
            mean = Σ_t (w_t × x_t)          加权均值，shape (B, D)
            diff = x - mean                  各时刻与均值的偏差
            var  = Σ_t (w_t × diff_t²)      加权方差
            std  = sqrt(var + 1e-8)          数值稳定的加权标准差

        Step 4: 拼接输出
            output = cat([mean, std], dim=-1)  shape: (B, 2D)

    参数:
        d_model: 特征维度 D
        alpha:   VAD 权重的初始值（可学习）
        beta:    QC 权重的初始值（可学习）

    输出: (B, 2×d_model)
    """
    """Attentive Statistics Pooling with VAD and quality control signals."""

    def __init__(self, d_model: int, alpha: float = 0.5, beta: float = 0.5) -> None:
        super().__init__()
        # 可学习的注意力投影：D → 1（每个时间步产生一个标量分数）
        self.attn = nn.Linear(d_model, 1)
        # α, β 作为 nn.Parameter 参与反向传播，会被优化器更新
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta = nn.Parameter(torch.tensor(beta))

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        vad: torch.Tensor,
        qc: torch.Tensor,
    ) -> torch.Tensor:
        """
        参数:
            x    : (B, T, D) — TCN 输出的时序特征
            mask : (B, T)    — bool 掩码，True=有效
            vad  : (B, T)    — 语音活动检测信号，float
            qc   : (B, T)    — 质量控制信号，float
        返回:
            (B, 2*D) — 均值和标准差的拼接
        """
        # Step 1: 计算注意力分数
        # self.attn(x): (B,T,D) → (B,T,1)，squeeze 后 → (B,T)
        e = self.attn(x).squeeze(-1)
        # 融合 VAD 和 QC：说话且高质量的帧获得更高权重
        # α, β 可学习，模型自动决定 VAD 和 QC 各贡献多少
        e = e + self.alpha * vad + self.beta * qc

        # Step 2: 掩码无效位置，做 Softmax 归一化
        # masked_fill：将 ~mask（无效位置）填充为 -inf，使 softmax 后权重=0
        e = e.masked_fill(~mask, float("-inf"))
        w = F.softmax(e, dim=-1)
        # 二次 mask：处理全掩码的极端情况（全 -inf → softmax 输出 NaN）
        w = w.masked_fill(~mask, 0.0)   # to avoid NaN in mean/std when all masked

        # Step 3: 计算加权统计量
        w_unsq = w.unsqueeze(-1)           # (B,T) → (B,T,1)，便于广播
        mean = (w_unsq * x).sum(dim=1)    # 加权均值 (B,D)

        # 加权标准差：衡量时序特征的离散程度
        diff = x - mean.unsqueeze(1)       # (B,T,D) - (B,1,D) = (B,T,D)
        var = (w_unsq * diff ** 2).sum(dim=1)   # 加权方差 (B,D)
        std = torch.sqrt(var.clamp(min=1e-8))   # clamp 避免 sqrt(负数)，数值稳定

        # Step 4: 拼接均值和标准差，输出 2D 维向量
        return torch.cat([mean, std], dim=-1)

class CrossModalAttention(nn.Module):
    """跨模态交叉注意力层

    Q 来自模态 A，K/V 来自模态 B → 模态 A 的每个时间步可以"关注"模态 B
    中最相关的时间步，获取互补信息。

    包含可学习门控：基于两模态全局统计量决定跨模态信息的混合比例，
    当互补模态缺失或质量低时自动退化为恒等映射。
    """

    def __init__(self, d_model: int, n_heads: int = 1, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.drop_attn = nn.Dropout(dropout)
        self.drop_out = nn.Dropout(dropout)

        # 门控：基于两模态全局统计量决定跨模态信息混合比例
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        query_mask: torch.Tensor,
        kv_mask: torch.Tensor,
    ) -> torch.Tensor:
        """返回 query 模态经跨模态注意力增强后的特征 (B, T_q, D)"""
        B, T_q, D = query.shape
        T_kv = key_value.size(1)

        residual = query

        # 计算门控值：基于两模态的全局均值池化
        q_pooled = query.sum(dim=1) / query_mask.float().sum(dim=1, keepdim=True).clamp(min=1)
        kv_pooled = key_value.sum(dim=1) / kv_mask.float().sum(dim=1, keepdim=True).clamp(min=1)
        gate_val = self.gate(torch.cat([q_pooled, kv_pooled], dim=-1))  # (B, 1)

        # Pre-Norm + 多头注意力
        q_norm = self.norm_q(query)
        kv_norm = self.norm_kv(key_value)

        Q = self.W_q(q_norm).view(B, T_q, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(kv_norm).view(B, T_kv, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(kv_norm).view(B, T_kv, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_mask = kv_mask.unsqueeze(1).unsqueeze(2)
        scores = scores.masked_fill(~attn_mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = weights.masked_fill(~attn_mask, 0.0)
        weights = self.drop_attn(weights)

        out = torch.matmul(weights, V)
        out = out.transpose(1, 2).contiguous().view(B, T_q, D)
        out = self.drop_out(self.W_o(out))

        # 门控残差：gate=0 时退化为恒等映射（原模态不变）
        out = residual + gate_val.unsqueeze(1) * out
        out = out * query_mask.unsqueeze(-1).float()
        return out


class BidirectionalCrossModalFusion(nn.Module):
    """双向跨模态融合

    同时让音频关注视频、视频关注音频，实现信息双向流动。
    """

    def __init__(self, d_model: int, n_heads: int = 1, dropout: float = 0.1):
        super().__init__()
        self.audio_attends_video = CrossModalAttention(d_model, n_heads, dropout)
        self.video_attends_audio = CrossModalAttention(d_model, n_heads, dropout)

    def forward(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
        mask_audio: torch.Tensor,
        mask_video: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 若一模态完全缺失，跳过该方向的跨模态注意力
        has_video = mask_video.any(dim=1)  # (B,)
        has_audio = mask_audio.any(dim=1)

        audio_enriched = audio
        video_enriched = video

        if has_video.any():
            audio_enriched = self.audio_attends_video(audio, video, mask_audio, mask_video)
        if has_audio.any():
            video_enriched = self.video_attends_audio(video, audio, mask_video, mask_audio)

        return audio_enriched, video_enriched


class MTCNBackbone(nn.Module):
    """多模态 TCN 骨干网络 (Multi-modal Temporal Convolutional Network Backbone)

    处理音频和视频多模态特征，输出会话级别的固定维度表示。

    ====================================================================
    整体数据流（以默认配置为例）
    ====================================================================

    输入 batch:
        audio_groups:        {名称: (B, T, D_in)}  — 音频时序特征，各 D_in 不同
        video_groups:        {名称: (B, T, D_in)}  — 视频时序特征
        audio_pooled_groups: {名称: (B, D_in)}     — 已全局池化的音频特征（无时序）
        mask_audio:          (B, T)  bool          — 音频有效帧掩码
        mask_video:          (B, T)  bool          — 视频有效帧掩码
        vad_signal:          (B, T)  float         — 语音活动检测
        qc_quality:          (B, T)  float         — 帧质量信号
        session_idx:         (B,)    int           — 会话索引（0 ~ n_sessions-1）

    Step 1: 特征适配（GroupAdapter）
        audio_adapted: [(B,T,64), (B,T,64), ...]   # 各音频特征 → d_adapter=64
        video_adapted: [(B,T,64), (B,T,64), ...]

    Step 2: 模态内融合（ModalityFusion = 拼接 + Linear）
        a: (B, T, 256)   # 音频各组拼接后投影到 d_model=256
        v: (B, T, 256)   # 视频同理

    Step 3: 掩码 + TCN 时序建模
        a = a × mask_audio    # 无效帧置零（防噪声进入 TCN）
        a = audio_tcn(a)      # 4层膨胀残差块，感受野 31 帧
        v = video_tcn(v)      # 同理

    Step 4: ASP 统计池化（变长 → 固定）
        z_a: (B, 512)    # 音频均值+标准差，2×d_model=512
        z_v: (B, 512)    # 视频同理

    Step 5: 收集全部特征
        parts = [z_a, z_v]                    # 时序特征统计量
                + [pooled_feat1, ...]          # 已池化音频特征（无时序）
                + [session_embed(idx)]         # 会话嵌入（捕获采集场景差异）

    Step 6: 最终 MLP 融合
        z = cat(parts)                         # (B, fusion_in)
        output = fusion_mlp(z)                 # (B, d_shared=256)

    ====================================================================
    模块组成
    ====================================================================

        audio_adapters:       ModuleDict，每个音频特征组一个 GroupAdapter
        audio_pooled_adapters: ModuleDict，池化特征用 Sequential（无时序需求）
        video_adapters:       ModuleDict，每个视频特征组一个 GroupAdapter
        audio_fusion:         ModalityFusion，音频模态内融合
        video_fusion:         ModalityFusion，视频模态内融合
        audio_tcn:            TCN，音频时序建模
        video_tcn:            TCN，视频时序建模
        audio_asp:            ASP，音频统计池化
        video_asp:            ASP，视频统计池化
        session_embed:        Embedding，会话嵌入
        fusion_mlp:           Sequential(Linear→GELU→Dropout→Linear)

    参数:
        cfg: BackboneConfig 配置对象

    输出:
        (B, d_shared): 会话级别表示，供下游任务头（TaskHead）使用
    """
    def __init__(self, cfg: BackboneConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # ── 特征适配层 ──────────────────────────────────────────────────
        # 每个特征组独立一个 GroupAdapter，因为各组特征维度不同
        # ModuleDict 保证参数被正确注册，key 是特征名（如 "wav2vec"）
        self.audio_adapters = nn.ModuleDict({
            name: GroupAdapter(d_in, cfg.d_adapter, cfg.dropout)
            for name, d_in in cfg.audio_group_dims.items()
        })
        # 池化特征（无时序维度）用更简单的 Sequential，不需要 GroupAdapter 的时序处理
        self.audio_pooled_adapters = nn.ModuleDict({
            name: nn.Sequential(
                nn.LayerNorm(d_in),
                nn.Linear(d_in, cfg.d_adapter),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
            )
            for name, d_in in cfg.audio_pooled_group_dims.items()
        })
        self.video_adapters = nn.ModuleDict({
            name: GroupAdapter(d_in, cfg.d_adapter, cfg.dropout)
            for name, d_in in cfg.video_group_dims.items()
        })
        # sorted 保证特征顺序确定性（字典遍历顺序在不同 Python 版本可能不同）
        self.audio_group_names = sorted(cfg.audio_group_dims.keys())
        self.audio_pooled_group_names = sorted(cfg.audio_pooled_group_dims.keys())
        self.video_group_names = sorted(cfg.video_group_dims.keys())

        # ── 模态内融合层 ────────────────────────────────────────────────
        # 音频：将 n 组 d_adapter 维特征融合为一个 d_model 维特征
        self.audio_fusion = ModalityFusion(
            len(self.audio_group_names), cfg.d_adapter, cfg.d_model
        )
        self.video_fusion = ModalityFusion(
            len(self.video_group_names), cfg.d_adapter, cfg.d_model
        )

        # ── 时序卷积网络 ─────────────────────────────────────────────────
        # 音频和视频各自独立的 TCN，因为两种模态的时序模式不同
        self.audio_tcn = TCN(cfg.d_model, cfg.tcn_layers, cfg.tcn_kernel_size, cfg.dropout)
        self.video_tcn = TCN(cfg.d_model, cfg.tcn_layers, cfg.tcn_kernel_size, cfg.dropout)

        # ── 跨模态融合（可选） ──────────────────────────────────────────────
        self.use_cross_modal = cfg.use_cross_modal
        if self.use_cross_modal:
            self.cross_modal = BidirectionalCrossModalFusion(
                cfg.d_model, cfg.cm_n_heads, cfg.dropout
            )

        # ── 注意力统计池化 ──────────────────────────────────────────────
        self.audio_asp = ASP(cfg.d_model, cfg.asp_alpha, cfg.asp_beta)
        self.video_asp = ASP(cfg.d_model, cfg.asp_alpha, cfg.asp_beta)

        # ── 计算最终融合层的输入维度 ─────────────────────────────────────
        # z_a: (B, 2×d_model)，z_v: (B, 2×d_model) → 2×2×d_model
        fusion_in = 2 * cfg.d_model * 2   # 音频ASP输出 + 视频ASP输出
        # 已全局池化的音频特征（每组 d_adapter 维）
        fusion_in += len(self.audio_pooled_group_names) * cfg.d_adapter
        # 会话嵌入维度
        fusion_in += cfg.d_session

        # ── 会话嵌入 ────────────────────────────────────────────────────
        # 将离散的会话索引（0,1,2,3）映射为连续向量
        # 作用：捕获不同录制场景、设备、环境的系统性差异
        # 例：某个会话因为麦克风型号不同，整体音质偏暗，嵌入可以学到这个偏差
        self.session_embed = nn.Embedding(cfg.n_sessions, cfg.d_session)

        # ── 最终融合 MLP ────────────────────────────────────────────────
        # 两层 MLP：融合所有模态信息，输出最终的会话级表示
        # 第一层：降维+激活（可能降维，如 1040 → 256）
        # 第二层：特征变换（保持 d_shared 维度）
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_in, cfg.d_shared),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_shared, cfg.d_shared),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """权重初始化

        Xavier 均匀初始化（适合 Linear 层）：
            W ~ Uniform(-√(6/(fan_in+fan_out)), +√(6/(fan_in+fan_out)))
            设计原则：使前向传播和反向传播的方差保持稳定
            适用场景：GELU/Tanh 等对称激活函数

        Embedding 用小标准差正态初始化（std=0.02）：
            来自 GPT/BERT 的经验做法，防止嵌入值过大破坏初始训练稳定性

        注意：权重归一化的卷积层有自己的初始化机制，不在此处处理。
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """前向传播

        参数:
            batch: 字典，包含以下键：
                "audio_groups":        {特征名: (B,T,D)} — 音频时序特征
                "video_groups":        {特征名: (B,T,D)} — 视频时序特征
                "audio_pooled_groups": {特征名: (B,D)}   — 已池化音频特征
                "mask_audio":          (B,T) bool        — 音频掩码
                "mask_video":          (B,T) bool        — 视频掩码
                "vad_signal":          (B,T) float       — 语音活动信号
                "qc_quality":          (B,T) float       — 质量控制信号
                "session_idx":         (B,)  int         — 会话索引

        返回:
            (B, d_shared) — 会话级特征表示
        """
        # ── Step 1: 特征适配 ─────────────────────────────────────────────
        # 每组特征通过对应的 GroupAdapter 投影到 d_adapter 维度
        # sorted 保证遍历顺序与 __init__ 一致
        audio_adapted = [
            self.audio_adapters[n](batch["audio_groups"][n])
            for n in self.audio_group_names
        ]
        video_adapted = [
            self.video_adapters[n](batch["video_groups"][n])
            for n in self.video_group_names
        ]

        # ── Step 2: 模态内融合 ───────────────────────────────────────────
        # 将列表中各组特征拼接后线性投影到 d_model 维
        a = self.audio_fusion(audio_adapted)   # (B, T, d_model)
        v = self.video_fusion(video_adapted)   # (B, T, d_model)

        # ── Step 3: 掩码 + TCN 时序建模 ─────────────────────────────────
        mask_a = batch["mask_audio"]           # (B, T) bool
        mask_v = batch["mask_video"]
        # 掩码置零：确保 padding 帧不影响 TCN 计算
        # unsqueeze(-1) 使 (B,T) 广播到 (B,T,d_model)
        a = a * mask_a.unsqueeze(-1).float()
        v = v * mask_v.unsqueeze(-1).float()

        a = self.audio_tcn(a, mask_a)          # 4层膨胀卷积，感受野逐层扩大
        v = self.video_tcn(v, mask_v)

        # ── Step 3.5: 跨模态交叉注意力（可选） ────────────────────────────
        if self.use_cross_modal:
            a, v = self.cross_modal(a, v, mask_a, mask_v)

        # ── Step 4: ASP 统计池化 ─────────────────────────────────────────
        # 利用 VAD 和 QC 信号引导注意力，压缩时序 → 固定大小统计量
        vad = batch["vad_signal"]              # (B, T)
        qc = batch["qc_quality"]               # (B, T)
        z_a = self.audio_asp(a, mask_a, vad, qc)   # (B, 2×d_model)
        z_v = self.video_asp(v, mask_v, vad, qc)   # (B, 2×d_model)

        # ── Step 5: 收集所有特征 ─────────────────────────────────────────
        parts = [z_a, z_v]                     # 时序模态统计量
        # 已池化的音频特征（无时序，直接用 Sequential 适配后拼接）
        parts.extend(
            self.audio_pooled_adapters[name](batch["audio_pooled_groups"][name])
            for name in self.audio_pooled_group_names
        )
        # 会话嵌入：学习不同录制会话的系统性偏差
        parts.append(self.session_embed(batch["session_idx"]))  # (B, d_session)

        # ── Step 6: 最终 MLP 融合 ────────────────────────────────────────
        z = torch.cat(parts, dim=-1)           # (B, fusion_in)
        return self.fusion_mlp(z)              # (B, d_shared)
