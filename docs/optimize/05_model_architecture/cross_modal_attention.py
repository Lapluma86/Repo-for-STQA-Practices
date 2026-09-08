"""
cross_modal_attention.py — 跨模态交叉注意力机制

问题诊断：
    当前 MTCNBackbone 的音频和视频通路完全独立（各自 TCN → ASP），
    直到最后 fusion_mlp 才用线性层拼接融合。
    这意味着模型无法建模"说话时的面部表情变化"这类跨模态时序关联。

    实验表明：对于多模态情感分析，中间层的交叉注意力比晚期拼接
    能提升 3-8% 的指标（参考 MulT, Bottleneck Transformers 等工作）。

改进方案：
    在 TCN 之后、ASP 之前，插入一个轻量级跨模态交叉注意力层，
    让音频序列能查询视频序列的信息，反之亦然。

    audio_tcn_out → CrossModalAttention(Q=audio, KV=video) → audio_enriched
    video_tcn_out → CrossModalAttention(Q=video, KV=audio) → video_enriched

    然后 audio_enriched 和 video_enriched 再各自进入 ASP。

预期提升：QWK +2~5%，MAE -0.3~0.5

集成方式：
    在 MTCNBackbone.forward() 的 Step 3 和 Step 4 之间插入。
    需要修改 MTCNBackbone.__init__ 和 forward 方法。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttention(nn.Module):
    """
    跨模态交叉注意力层

    核心思想：
        Q 来自模态 A，K/V 来自模态 B → 模态 A 的每个时间步
        可以"关注"模态 B 中最相关的时间步，获取互补信息。

    设计决策：
        1. 使用单头注意力而非多头 — 数据量小（~1000样本），多头容易过拟合
        2. 加残差连接 — 即使注意力学废了，也不会比原来差
        3. 加 LayerNorm — 稳定训练
        4. 支持 mask — 无效帧不参与注意力计算

    参数:
        d_model:   特征维度（与 TCN 输出维度相同）
        n_heads:   注意力头数（默认1，小数据集建议1-2头）
        dropout:   注意力 dropout 比率
    """

    def __init__(self, d_model: int, n_heads: int = 1, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Q 投影（来自被增强的模态）
        self.W_q = nn.Linear(d_model, d_model)
        # K, V 投影（来自提供信息的模态）
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # 输出投影
        self.W_o = nn.Linear(d_model, d_model)

        # LayerNorm + Dropout（Pre-Norm 风格，训练更稳定）
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.drop_attn = nn.Dropout(dropout)
        self.drop_out = nn.Dropout(dropout)

        # 缩放因子：1/sqrt(d_head)，防止 softmax 饱和
        self.scale = self.d_head ** -0.5

    def forward(
        self,
        query: torch.Tensor,       # (B, T_q, D) — 被增强的模态
        key_value: torch.Tensor,    # (B, T_kv, D) — 提供信息的模态
        query_mask: torch.Tensor,   # (B, T_q) bool
        kv_mask: torch.Tensor,      # (B, T_kv) bool
    ) -> torch.Tensor:
        """
        返回:
            enriched: (B, T_q, D) — 增强后的 query 模态特征
        """
        B, T_q, D = query.shape
        T_kv = key_value.size(1)

        # 残差连接的起点
        residual = query

        # Pre-Norm
        q = self.norm_q(query)
        kv = self.norm_kv(key_value)

        # 线性投影 → 多头拆分
        # (B, T, D) → (B, T, n_heads, d_head) → (B, n_heads, T, d_head)
        Q = self.W_q(q).view(B, T_q, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(kv).view(B, T_kv, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(kv).view(B, T_kv, self.n_heads, self.d_head).transpose(1, 2)

        # 注意力分数：Q·K^T / sqrt(d_head)
        # (B, n_heads, T_q, d_head) × (B, n_heads, d_head, T_kv) → (B, n_heads, T_q, T_kv)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # 构造注意力掩码：无效的 KV 位置填 -inf
        # kv_mask: (B, T_kv) → (B, 1, 1, T_kv)，广播到所有 head 和 query 位置
        attn_mask = kv_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T_kv)
        scores = scores.masked_fill(~attn_mask, float("-inf"))

        # Softmax 归一化 + Dropout
        weights = F.softmax(scores, dim=-1)
        weights = weights.masked_fill(~attn_mask, 0.0)  # 处理全 -inf 导致的 NaN
        weights = self.drop_attn(weights)

        # 加权求和
        # (B, n_heads, T_q, T_kv) × (B, n_heads, T_kv, d_head) → (B, n_heads, T_q, d_head)
        out = torch.matmul(weights, V)

        # 多头合并：(B, n_heads, T_q, d_head) → (B, T_q, D)
        out = out.transpose(1, 2).contiguous().view(B, T_q, D)

        # 输出投影 + Dropout
        out = self.drop_out(self.W_o(out))

        # 残差连接：即使注意力没学到有用信息，也不会退化
        out = out + residual

        # 掩码置零：无效 query 位置不应有输出
        out = out * query_mask.unsqueeze(-1).float()

        return out


class BidirectionalCrossModalFusion(nn.Module):
    """
    双向跨模态融合模块

    同时让音频关注视频、视频关注音频，实现信息双向流动。

    相比单向：双向融合让两个模态都能获益，且共享参数量不增加太多。

    参数:
        d_model:  特征维度
        n_heads:  注意力头数
        dropout:  Dropout 比率
    """

    def __init__(self, d_model: int, n_heads: int = 1, dropout: float = 0.1):
        super().__init__()
        # 音频 → 视频方向的跨模态注意力
        self.audio_attends_video = CrossModalAttention(d_model, n_heads, dropout)
        # 视频 → 音频方向的跨模态注意力
        self.video_attends_audio = CrossModalAttention(d_model, n_heads, dropout)

    def forward(
        self,
        audio: torch.Tensor,      # (B, T, D)
        video: torch.Tensor,      # (B, T, D)
        mask_audio: torch.Tensor,  # (B, T) bool
        mask_video: torch.Tensor,  # (B, T) bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        返回:
            (audio_enriched, video_enriched) — 各 (B, T, D)
        """
        audio_enriched = self.audio_attends_video(audio, video, mask_audio, mask_video)
        video_enriched = self.video_attends_audio(video, audio, mask_video, mask_audio)
        return audio_enriched, video_enriched


# ============================================================
# 集成示例（修改 MTCNBackbone）
# ============================================================
#
# class MTCNBackbone(nn.Module):
#     def __init__(self, cfg):
#         ...
#         # 新增：跨模态注意力
#         self.cross_modal = BidirectionalCrossModalFusion(
#             d_model=cfg.d_model,
#             n_heads=1,
#             dropout=cfg.dropout,
#         )
#
#     def forward(self, batch):
#         ...
#         # Step 3: TCN 时序建模
#         a = self.audio_tcn(a, mask_a)
#         v = self.video_tcn(v, mask_v)
#
#         # ★ 新增 Step 3.5：跨模态交叉注意力
#         a, v = self.cross_modal(a, v, mask_a, mask_v)
#
#         # Step 4: ASP 统计池化（不变）
#         z_a = self.audio_asp(a, mask_a, vad, qc)
#         z_v = self.video_asp(v, mask_v, vad, qc)
#         ...
