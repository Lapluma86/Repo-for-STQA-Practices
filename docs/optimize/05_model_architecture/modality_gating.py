"""
modality_gating.py — 自适应模态门控机制

问题诊断：
    当前的 fusion_mlp 对音频和视频 ASP 输出做简单拼接+线性融合。
    隐含假设：两个模态始终同等重要。

    但实际场景中：
    - 某些参与者的音频质量差（背景噪声、录音故障）→ 应降低音频权重
    - 某些参与者的视频质量差（遮挡、光照不均）→ 应降低视频权重
    - 不同评估项目对不同模态的依赖度不同
      （如"说话音量"更依赖音频，"面部表情"更依赖视频）

改进方案：
    在 fusion_mlp 之前，添加一个门控网络（Gating Network），
    根据两个模态的质量/内容自适应地计算融合权重。

    gate = σ(W_g · [z_a; z_v])  ∈ [0, 1]
    fused = gate × z_a + (1 - gate) × z_v

    门控网络从数据中学习"什么时候信任音频、什么时候信任视频"。

预期提升：QWK +1~3%，单模态缺失时鲁棒性显著提升

集成方式：
    替换 MTCNBackbone 的 fusion_mlp 部分。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ModalityGating(nn.Module):
    """
    自适应模态门控融合

    核心设计：
        1. 用两个模态的统计量（ASP 输出）共同决定融合权重
        2. Sigmoid 门控 → 输出在 [0, 1]，可解释为"对音频的信任度"
        3. 残差设计：gate=0.5 时退化为均值融合，不会比简单方案差

    为什么不用 attention（如 Transformer）做门控？
        - 门控只需要一个标量权重，参数量极小（~2D 参数）
        - 注意力机制对小数据集来说过于灵活，容易过拟合
        - 门控可以在 batch 内独立计算，无序列长度依赖

    参数:
        d_in:    单个模态的 ASP 输出维度（2 × d_model）
        d_gate:  门控网络隐层维度（默认 64，足够学习简单的质量判断）
    """

    def __init__(self, d_in: int, d_gate: int = 64):
        super().__init__()
        # 门控网络：接收两个模态拼接后的特征，输出门控值
        # 结构：Linear → ReLU → Linear → Sigmoid
        self.gate_net = nn.Sequential(
            nn.Linear(d_in * 2, d_gate),
            nn.ReLU(),
            nn.Linear(d_gate, d_in),
            nn.Sigmoid(),   # 输出 [0, 1]，可解释为"音频信任度"
        )

    def forward(
        self,
        z_audio: torch.Tensor,   # (B, d_in) — 音频 ASP 输出
        z_video: torch.Tensor,   # (B, d_in) — 视频 ASP 输出
    ) -> torch.Tensor:
        """
        返回:
            fused: (B, d_in) — 门控融合后的特征
        """
        # 拼接两个模态的统计量，让门控网络"看到"双方信息
        combined = torch.cat([z_audio, z_video], dim=-1)  # (B, 2*d_in)

        # 计算门控值：每个特征维度独立的融合权重
        gate = self.gate_net(combined)  # (B, d_in)，每个值在 [0, 1]

        # 加权融合：gate 高 → 偏向音频，gate 低 → 偏向视频
        fused = gate * z_audio + (1.0 - gate) * z_video

        return fused


class GatedFusionMLP(nn.Module):
    """
    门控融合 + MLP 的完整替换模块

    用于替换原始 MTCNBackbone 中的 fusion_mlp，
    在拼接之前先做门控融合，再与池化特征和会话嵌入一起投影。

    原始流程:
        parts = [z_a, z_v, pooled..., session_embed]
        z = cat(parts) → fusion_mlp → (B, d_shared)

    新流程:
        z_fused = ModalityGating(z_a, z_v)           # 门控融合
        parts = [z_fused, pooled..., session_embed]   # 注意维度变了
        z = cat(parts) → fusion_mlp → (B, d_shared)

    优势：
        - 融合在"知道两个模态各自质量"的基础上进行
        - 减少 fusion_mlp 的输入维度（2*d_in → d_in），降低过拟合风险
        - 门控值可以可视化，便于分析模型行为

    参数:
        d_asp:      ASP 输出维度（2 × d_model）
        d_pooled:   池化特征总维度（n_pooled_groups × d_adapter）
        d_session:  会话嵌入维度
        d_shared:   最终输出维度
        dropout:    Dropout 比率
    """

    def __init__(
        self,
        d_asp: int,
        d_pooled: int,
        d_session: int,
        d_shared: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.gating = ModalityGating(d_in=d_asp)

        # 融合后的输入维度：门控输出 + 池化特征 + 会话嵌入
        fusion_in = d_asp + d_pooled + d_session
        self.mlp = nn.Sequential(
            nn.Linear(fusion_in, d_shared),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_shared, d_shared),
        )

    def forward(
        self,
        z_audio: torch.Tensor,
        z_video: torch.Tensor,
        pooled_features: list[torch.Tensor],
        session_embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        参数:
            z_audio:         (B, d_asp) — 音频 ASP 输出
            z_video:         (B, d_asp) — 视频 ASP 输出
            pooled_features: list of (B, d_adapter) — 池化特征
            session_embed:   (B, d_session) — 会话嵌入

        返回:
            (B, d_shared) — 最终会话级表示
        """
        # 门控融合
        z_fused = self.gating(z_audio, z_video)

        # 收集所有特征
        parts = [z_fused] + pooled_features + [session_embed]
        z = torch.cat(parts, dim=-1)

        return self.mlp(z)


# ============================================================
# 集成示例
# ============================================================
#
# class MTCNBackbone(nn.Module):
#     def __init__(self, cfg):
#         ...
#         # 替换原始 fusion_mlp
#         d_asp = 2 * cfg.d_model
#         d_pooled = len(self.audio_pooled_group_names) * cfg.d_adapter
#         self.gated_fusion = GatedFusionMLP(
#             d_asp=d_asp,
#             d_pooled=d_pooled,
#             d_session=cfg.d_session,
#             d_shared=cfg.d_shared,
#             dropout=cfg.dropout,
#         )
#         # 删除原来的 self.fusion_mlp
#
#     def forward(self, batch):
#         ...
#         # Step 4: ASP
#         z_a = self.audio_asp(a, mask_a, vad, qc)
#         z_v = self.video_asp(v, mask_v, vad, qc)
#
#         # Step 5: 收集池化特征
#         pooled = [
#             self.audio_pooled_adapters[n](batch["audio_pooled_groups"][n])
#             for n in self.audio_pooled_group_names
#         ]
#         sess_emb = self.session_embed(batch["session_idx"])
#
#         # Step 6: 门控融合 + MLP（替换原始拼接+fusion_mlp）
#         return self.gated_fusion(z_a, z_v, pooled, sess_emb)
