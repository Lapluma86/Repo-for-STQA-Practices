"""
本模块在会话级骨干网络之上，增加了"参与者级"聚合能力。

背景：
    MTCNBackbone 处理的是单次会话（session）的特征，输出 (B, d_shared)。
    但实际场景中，同一个参与者（participant）会经历多次会话（如4次），
    最终的心理评估分数是对整个参与者的评估，需要跨会话聚合。

数据组织方式（flat batch）：
    输入数据将 n_participants × 4 个会话展平（flatten）成一个大 batch，
    骨干网络一次处理所有会话，得到 (n_participants×4, d_shared) 的会话表示，
    再 reshape 为 (n_participants, 4, d_shared) 后做跨会话聚合。

模块组成：
    ParticipantAggregator  — 将4个会话表示聚合为1个参与者表示
    SessionTypeClassifier  — 辅助任务：预测会话类型（用于多任务学习）
    GroupedModel           — 完整模型：骨干 + 聚合 + 辅助头
    CORALHead              — 改进的序数回归头（可学习的有序阈值间距）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mtcn_backbone import MTCNBackbone, BackboneConfig


class ParticipantAggregator(nn.Module):
    """
    跨会话聚合器：将同一参与者的多个会话表示聚合为一个参与者级表示。

    支持三种聚合方式：
        "mean"      — 有效会话的均值池化，再线性投影。最简单，无额外参数。
        "mlp"       — 均值池化后接两层MLP变换。适合需要非线性融合的情况。
        "attention" — 可学习注意力权重池化，让模型自动决定哪次会话更重要。

    为什么需要处理"有效会话"（session_valid）？
        某些参与者可能只完成了部分会话（如因故缺席），
        无效会话对应的特征是 padding，不应参与聚合计算。

    参数:
        d_in:   输入维度（单个会话表示的维度，即 d_shared）
        d_out:  输出维度（参与者表示维度）
        method: 聚合方式，"mean" / "mlp" / "attention"
        dropout: MLP方式下的 Dropout 比率
    """

    def __init__(self, d_in: int, d_out: int, method: str = "mlp", dropout: float = 0.2):
        super().__init__()
        self.method = method
        self.d_in = d_in
        self.d_out = d_out

        if method == "mlp":
            # 均值池化 + 两层MLP，提供非线性融合能力
            self.mlp = nn.Sequential(
                nn.Linear(d_in, d_out),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_out, d_out),
            )
        elif method == "attention":
            # 可学习注意力：query 把每个会话表示压缩为一个标量分数
            self.query = nn.Linear(d_in, 1)
            self.proj = nn.Linear(d_in, d_out)
        elif method == "mean":
            # d_in == d_out 时用 Identity 节省参数
            self.proj = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    def forward(self, session_reprs: torch.Tensor, session_valid: torch.Tensor) -> torch.Tensor:
        """
        参数:
            session_reprs: (B, n_sessions, d_in)  — 各参与者的会话表示
            session_valid: (B, n_sessions) bool   — True 表示该会话有效
        返回:
            (B, d_out) — 参与者级别表示
        """
        # 无效会话置零，防止 padding 特征干扰聚合结果
        mask = session_valid.float().unsqueeze(-1)   # (B, n_sessions, 1)
        masked_reprs = session_reprs * mask

        if self.method == "mean":
            # 有效会话数量（clamp(min=1) 防止除零，处理全部无效的边界情况）
            n_valid = mask.sum(dim=1).clamp(min=1)   # (B, 1)
            pooled = masked_reprs.sum(dim=1) / n_valid  # 均值池化 (B, d_in)
            return self.proj(pooled)

        elif self.method == "mlp":
            n_valid = mask.sum(dim=1).clamp(min=1)
            pooled = masked_reprs.sum(dim=1) / n_valid
            return self.mlp(pooled)

        elif self.method == "attention":
            # 计算每个会话的注意力分数
            scores = self.query(session_reprs).squeeze(-1)   # (B, n_sessions)
            # 无效会话填充 -inf，softmax 后权重趋近于 0
            scores = scores.masked_fill(~session_valid, float("-inf"))
            weights = F.softmax(scores, dim=-1)              # (B, n_sessions)
            # 再次 mask 防止全无效时产生 NaN
            weights = weights.masked_fill(~session_valid, 0.0)
            # 加权求和：(B, n_sessions, 1) × (B, n_sessions, d_in) → (B, d_in)
            pooled = (weights.unsqueeze(-1) * session_reprs).sum(dim=1)
            return self.proj(pooled)


class SessionTypeClassifier(nn.Module):
    """
    会话类型分类器（辅助任务头）

    用途：多任务学习中的辅助损失，预测每个会话的类型（如不同的录制场景/条件）。
    辅助损失的好处：给骨干网络额外的监督信号，引导它学习更有区分度的会话表示。

    结构：Linear → GELU → Linear（轻量两层MLP）
    输入：单个会话表示 (B×n_sessions, d_in)
    输出：会话类型 logit (B×n_sessions, n_classes)
    """

    def __init__(self, d_in: int, n_classes: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_in, 64),
            nn.GELU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class GroupedModel(nn.Module):
    """
    完整的分组模型：骨干网络 + 跨会话聚合 + 辅助分类头

    数据流：
        flat_batch（展平的所有会话数据）
            ↓ backbone（MTCNBackbone）
        session_reprs: (n_participants×4, d_shared)   — 每个会话的表示
            ↓ reshape
        session_grid: (n_participants, 4, d_shared)   — 按参与者分组
            ↓ aggregator（ParticipantAggregator）
        participant_repr: (n_participants, d_shared)  — 参与者级表示（供任务头用）
            ↓ session_type_head（辅助任务，作用于原始 session_reprs）
        session_type_logits: (n_participants×4, 4)   — 会话类型预测

    参数:
        backbone:           预构建的 MTCNBackbone 实例
        d_shared:           骨干网络输出维度（同时也是聚合器 I/O 维度）
        aggregator_method:  跨会话聚合方式，"mean" / "mlp" / "attention"
        dropout:            聚合器 Dropout 比率
    """

    def __init__(
        self,
        backbone: MTCNBackbone,
        d_shared: int,
        aggregator_method: str = "mlp",
        dropout: float = 0.2,
        aux_encoder=None,
        aux_heads=None,
    ):
        super().__init__()
        self.backbone = backbone
        self.aggregator = ParticipantAggregator(
            d_in=d_shared, d_out=d_shared,
            method=aggregator_method, dropout=dropout,
        )
        self.session_type_head = SessionTypeClassifier(d_in=d_shared)
        self.aux_encoder = aux_encoder
        self.aux_heads = aux_heads

    def forward(
        self,
        flat_batch: dict,
        n_participants: int,
        session_valid: torch.Tensor,
        aux_attrs: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        参数:
            flat_batch:     展平的 batch 字典，包含 n_participants×4 个会话的特征
            n_participants: 参与者数量 B
            session_valid:  (B, 4) bool，标记每个参与者的各会话是否有效
            aux_attrs:      (B, 5) 辅助属性，可选
        返回:
            dict，包含：
                "session_reprs":      (B×4, d_shared) — 各会话的骨干输出
                "participant_repr":   (B, d_shared + aux_dim)   — 聚合后的参与者表示（可能拼接了辅助属性）
                "session_type_logits":(B×4, 4)       — 辅助任务预测
        """
        # 骨干网络一次处理所有会话（利用 batch 并行加速）
        session_reprs = self.backbone(flat_batch)   # (B×4, d_shared)

        B = n_participants
        # 将展平的会话表示重新分组：(B×4, d_shared) → (B, 4, d_shared)
        # 此处硬编码 4 表示每个参与者固定 4 次会话
        session_grid = session_reprs.view(B, 4, -1)

        # 跨会话聚合：(B, 4, d_shared) + valid_mask → (B, d_shared)
        participant_repr = self.aggregator(session_grid, session_valid)

        # LUPI: 从纯音视频表示预测辅助属性，提供额外监督信号
        aux_logits = None
        if self.aux_heads is not None:
            aux_logits = self.aux_heads(participant_repr)

        # 如果启用辅助属性，编码并拼接到参与者表示
        if self.aux_encoder is not None and aux_attrs is not None:
            aux_encoded = self.aux_encoder(aux_attrs)  # (B, aux_dim)
            participant_repr = torch.cat([participant_repr, aux_encoded], dim=-1)  # (B, d_shared + aux_dim)

        # 辅助任务：在单会话粒度预测会话类型（用于多任务学习）
        session_type_logits = self.session_type_head(session_reprs)   # (B×4, 4)

        return {
            "session_reprs": session_reprs,
            "participant_repr": participant_repr,
            "session_type_logits": session_type_logits,
            "aux_logits": aux_logits,
        }


class CORALHead(nn.Module):
    """
    CORAL 序数回归头（Consistent Rank Logits）

    相比 A2OrdinalHead 的改进：
        A2OrdinalHead：为每个项目的每个阈值独立输出 logit，
                        不保证阈值间的有序性（阈值1可能比阈值2更难通过）。
        CORALHead：    显式建模有序阈值间距，保证阈值天然单调递增。

    核心设计（有序阈值构造）：
        Step 1：为每个项目预测一个连续分数 score（公共 feature）
        Step 2：为每个项目学习 n_thresholds 个间距（spacing），用 softplus 保证正数
        Step 3：用累积和（cumsum）将间距转为单调递增的阈值序列
                thresholds[k] = Σ spacing[0..k]，天然满足 t1 < t2 < t3
        Step 4：logit[k] = score - thresholds[k]
                score 高于阈值k → logit>0 → P(score>=k+1) > 0.5

    为什么这样设计更好？
        所有阈值共享同一个 score，减少参数量（n_items 个线性输出 vs n_items×n_thresholds）。
        cumsum 结构天然保证阈值单调性，无需后处理强制约束。

    参数:
        d_in:         输入维度
        n_items:      评估项目数（默认21）
        n_thresholds: 阈值数（默认3，对应0-3共4个等级）
    """

    def __init__(self, d_in: int, n_items: int = 21, n_thresholds: int = 3):
        super().__init__()
        self.n_items = n_items
        self.n_thresholds = n_thresholds

        # 每个项目一个连续分数（共享 feature，比 A2OrdinalHead 参数量少）
        self.score_fc = nn.Linear(d_in, n_items)

        # 可学习的原始阈值间距参数，初始化为 0.5（softplus(0.5)≈0.97，间距约为1）
        # shape: (n_items, n_thresholds)，每个项目独立学习自己的阈值间距
        self.raw_thresholds = nn.Parameter(torch.zeros(n_items, n_thresholds))
        nn.init.constant_(self.raw_thresholds, 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: (B, d_in)
        返回:
            logits: (B, n_items, n_thresholds)
        """
        # 每个项目的连续分数：(B, n_items)
        scores = self.score_fc(x)

        # softplus 保证间距为正数（softplus(x) = log(1+e^x) > 0）
        spacings = F.softplus(self.raw_thresholds)       # (n_items, n_thresholds)
        # 累积求和：将正数间距转为单调递增的阈值
        # 例：spacings=[0.5, 0.8, 0.6] → thresholds=[0.5, 1.3, 1.9]
        thresholds = torch.cumsum(spacings, dim=-1)      # (n_items, n_thresholds)

        # 广播相减：score - threshold，分数高于阈值则 logit > 0
        # scores.unsqueeze(-1): (B, n_items, 1)
        # thresholds.unsqueeze(0): (1, n_items, n_thresholds)
        logits = scores.unsqueeze(-1) - thresholds.unsqueeze(0)  # (B, n_items, n_thresholds)
        return logits

    # 以下三种解码方式与 A2OrdinalHead 完全相同，见 heads.py 注释
    @staticmethod
    def predict_int(logits: torch.Tensor) -> torch.Tensor:
        """简单解码：各阈值独立判断后求和（可能产生非单调结果）"""
        return (torch.sigmoid(logits) > 0.5).long().sum(dim=-1)

    @staticmethod
    def predict_int_monotonic(logits: torch.Tensor) -> torch.Tensor:
        """单调解码（推荐）：强制累积概率 p1>=p2>=p3，取概率最大等级"""
        s = torch.sigmoid(logits)
        p1 = s[..., 0]
        p2 = torch.min(s[..., 1], p1)
        p3 = torch.min(s[..., 2], p2)
        P0 = 1.0 - p1
        P1 = p1 - p2
        P2 = p2 - p3
        P3 = p3
        class_probs = torch.stack([P0, P1, P2, P3], dim=-1)
        return class_probs.argmax(dim=-1)

    @staticmethod
    def predict_expectation(logits: torch.Tensor) -> torch.Tensor:
        """期望解码：E[score] = p1 + p2 + p3，四舍五入取整"""
        s = torch.sigmoid(logits)
        p1 = s[..., 0]
        p2 = torch.min(s[..., 1], p1)
        p3 = torch.min(s[..., 2], p2)
        E = p1 + p2 + p3
        return E.round().long().clamp(0, 3)
