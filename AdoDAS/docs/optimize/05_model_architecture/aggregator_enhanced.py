"""
aggregator_enhanced.py — 增强的跨会话聚合器

问题诊断：
    当前 ParticipantAggregator 的 attention 模式使用单个线性层（d_in → 1）
    计算注意力分数。这只能基于"会话表示的整体相似度"来分配权重，
    无法建模以下关键信息：

    1. 会话类型差异：A01（自由对话）和 B01-B03（结构化任务）的信息量不同
    2. 会话间的关系：某两次会话表现一致 vs 表现矛盾，对诊断有不同意义
    3. 会话质量差异：某次会话因设备问题数据质量差，应降低权重

改进方案 1 — 条件注意力聚合器：
    引入会话类型嵌入作为注意力的额外条件，让模型学会
    "A01 类型的会话应该获得多大权重"。

改进方案 2 — 多头注意力聚合器：
    使用多头自注意力建模会话间的关系，再用 [CLS] token 或
    均值池化得到参与者表示。

预期提升：QWK +1~3%
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalAttentionAggregator(nn.Module):
    """
    条件注意力聚合器：结合会话类型信息的加权聚合

    核心改进：
        原始：score = Linear(session_repr)   — 仅基于内容
        改进：score = Linear([session_repr; type_embed])  — 基于内容+类型

    为什么会话类型信息重要？
        - A01（自由对话）：自然行为，信息量大但噪声也多
        - B01-B03（结构化任务）：标准化刺激，信噪比高但覆盖面窄
        - 模型需要学会"综合不同角度的观察"而非"简单平均"

    参数:
        d_in:       输入维度（会话表示维度，即 d_shared）
        d_out:      输出维度（参与者表示维度）
        n_sessions: 会话类型数量（默认 4）
        d_type:     类型嵌入维度（默认 16）
        dropout:    Dropout 比率
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        n_sessions: int = 4,
        d_type: int = 16,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out

        # 会话类型嵌入：学习每种会话类型的固有权重偏好
        self.type_embed = nn.Embedding(n_sessions, d_type)

        # 注意力打分：基于内容+类型
        self.query = nn.Sequential(
            nn.Linear(d_in + d_type, d_in // 4),
            nn.Tanh(),        # Tanh 限制分数范围，防止注意力过于尖锐
            nn.Linear(d_in // 4, 1),
        )

        # 输出投影
        self.proj = nn.Linear(d_in, d_out)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        session_reprs: torch.Tensor,    # (B, n_sessions, d_in)
        session_valid: torch.Tensor,     # (B, n_sessions) bool
        session_types: torch.Tensor | None = None,  # (B, n_sessions) int，可选
    ) -> torch.Tensor:
        """
        参数:
            session_reprs: (B, n_sessions, d_in)
            session_valid: (B, n_sessions) bool
            session_types: (B, n_sessions) int — 每个会话的类型索引
                           如果为 None，退化为按顺序编号 [0,1,2,3]

        返回:
            (B, d_out) — 参与者级表示
        """
        B, S, D = session_reprs.shape

        # 生成会话类型索引（如果未提供，按默认顺序）
        if session_types is None:
            session_types = torch.arange(S, device=session_reprs.device).unsqueeze(0).expand(B, -1)

        # 获取类型嵌入：(B, S, d_type)
        type_emb = self.type_embed(session_types)

        # 拼接内容和类型信息：(B, S, d_in + d_type)
        combined = torch.cat([session_reprs, type_emb], dim=-1)

        # 计算注意力分数：(B, S, 1) → (B, S)
        scores = self.query(combined).squeeze(-1)

        # 无效会话掩码
        scores = scores.masked_fill(~session_valid, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = weights.masked_fill(~session_valid, 0.0)

        # 加权聚合
        pooled = (weights.unsqueeze(-1) * session_reprs).sum(dim=1)  # (B, d_in)
        pooled = self.drop(pooled)

        return self.proj(pooled)


class SelfAttentionAggregator(nn.Module):
    """
    多头自注意力聚合器：建模会话间关系后聚合

    核心思想：
        不仅考虑"每个会话有多重要"，还考虑"会话之间的关系"。
        例：如果某次会话的表现与其他三次都不一致（离群），
        自注意力可以识别这种模式并降低其影响。

    工作流程：
        1. 为每个会话添加位置/类型信息
        2. 多头自注意力让会话相互"对话"
        3. 对增强后的会话表示做加权池化

    参数:
        d_in:       输入维度
        d_out:      输出维度
        n_heads:    注意力头数（默认 2）
        n_sessions: 会话数量
        dropout:    Dropout 比率
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        n_heads: int = 2,
        n_sessions: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()

        # 位置嵌入：为4个会话位置各学习一个向量
        self.pos_embed = nn.Embedding(n_sessions, d_in)

        # 多头自注意力
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_in,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,  # 输入格式 (B, S, D)
        )

        # 前馈网络（标准 Transformer 结构）
        self.ffn = nn.Sequential(
            nn.Linear(d_in, d_in * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_in * 2, d_in),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_in)
        self.norm2 = nn.LayerNorm(d_in)

        # 最终聚合的注意力打分
        self.pool_query = nn.Linear(d_in, 1)
        self.proj = nn.Linear(d_in, d_out)

    def forward(
        self,
        session_reprs: torch.Tensor,    # (B, n_sessions, d_in)
        session_valid: torch.Tensor,     # (B, n_sessions) bool
    ) -> torch.Tensor:
        """
        返回:
            (B, d_out) — 参与者级表示
        """
        B, S, D = session_reprs.shape

        # 添加位置信息
        pos_idx = torch.arange(S, device=session_reprs.device)
        x = session_reprs + self.pos_embed(pos_idx).unsqueeze(0)

        # 构造 key_padding_mask：True 表示被忽略的位置
        # nn.MultiheadAttention 的 mask 语义与我们的 session_valid 相反
        key_padding_mask = ~session_valid  # True = 无效 = 被忽略

        # 自注意力 + 残差
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, key_padding_mask=key_padding_mask)
        x = x + attn_out

        # 前馈 + 残差
        x = x + self.ffn(self.norm2(x))

        # 掩码置零
        x = x * session_valid.unsqueeze(-1).float()

        # 注意力池化
        scores = self.pool_query(x).squeeze(-1)  # (B, S)
        scores = scores.masked_fill(~session_valid, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = weights.masked_fill(~session_valid, 0.0)

        pooled = (weights.unsqueeze(-1) * x).sum(dim=1)  # (B, d_in)
        return self.proj(pooled)


# ============================================================
# 集成示例（替换 GroupedModel 中的 aggregator）
# ============================================================
#
# from docs.optimize.model_architecture.aggregator_enhanced import (
#     ConditionalAttentionAggregator,
#     SelfAttentionAggregator,
# )
#
# class GroupedModel(nn.Module):
#     def __init__(self, backbone, d_shared, aggregator_method="conditional", ...):
#         ...
#         if aggregator_method == "conditional":
#             self.aggregator = ConditionalAttentionAggregator(
#                 d_in=d_shared, d_out=d_shared, dropout=dropout,
#             )
#         elif aggregator_method == "self_attention":
#             self.aggregator = SelfAttentionAggregator(
#                 d_in=d_shared, d_out=d_shared, dropout=dropout,
#             )
#
#     def forward(self, flat_batch, n_participants, session_valid):
#         ...
#         # 跨会话聚合（条件注意力版本，额外传入 session_types）
#         participant_repr = self.aggregator(
#             session_grid, session_valid,
#             session_types=session_types,  # (B, 4) — 从 batch 中获取
#         )
