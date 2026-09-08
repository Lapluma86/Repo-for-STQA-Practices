"""
辅助属性编码器

将五个分类辅助属性编码为固定维度的向量表示：
1. 家庭结构 (Family structure) - 6个类别 (1-6)
2. 独生子女 (Only child status) - 2个类别 (0-1)
3. 父母偏爱 (Parental favoritism) - 3个类别 (1-3)
4. 成绩变动 (Academic performance change) - 3个类别 (1-3)
5. 情绪变动 (Emotional state change) - 3个类别 (1-3)

设计要点：
- 使用embedding层对每个属性进行编码
- 处理缺失值（用特殊索引0表示缺失）
- 将所有编码拼接成固定维度输出
"""
import torch
import torch.nn as nn


class AuxiliaryAttributeEncoder(nn.Module):
    """
    辅助属性编码器

    参数:
        embed_dim: 每个属性的embedding维度
        dropout: Dropout比率
    """

    def __init__(self, embed_dim: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim

        # 为每个属性创建embedding层
        # num_embeddings = 类别数 + 1（为缺失值预留索引0）
        self.family_structure_embed = nn.Embedding(7, embed_dim)  # 6类 + 缺失
        self.only_child_embed = nn.Embedding(3, embed_dim)  # 2类 + 缺失
        self.parental_favoritism_embed = nn.Embedding(4, embed_dim)  # 3类 + 缺失
        self.grade_change_embed = nn.Embedding(4, embed_dim)  # 3类 + 缺失
        self.mood_change_embed = nn.Embedding(4, embed_dim)  # 3类 + 缺失

        self.dropout = nn.Dropout(dropout)

        # 输出维度 = 5个属性 × embed_dim
        self.output_dim = 5 * embed_dim

    def forward(self, aux_attrs: torch.Tensor) -> torch.Tensor:
        """
        参数:
            aux_attrs: (B, 5) 辅助属性张量，每列对应一个属性
                      缺失值用-1表示
        返回:
            (B, output_dim) 编码后的向量
        """
        # 将缺失值(-1)和无效值映射到索引0
        # 有效值映射到索引1, 2, 3, ...
        aux_attrs = aux_attrs.long()

        # 处理每个属性
        family_idx = torch.clamp(aux_attrs[:, 0], min=0, max=6)  # 0=缺失, 1-6=有效类别
        only_child_idx = torch.clamp(aux_attrs[:, 1], min=0, max=2)  # 0=缺失, 1-2=有效类别
        parental_fav_idx = torch.clamp(aux_attrs[:, 2], min=0, max=3)  # 0=缺失, 1-3=有效类别
        grade_change_idx = torch.clamp(aux_attrs[:, 3], min=0, max=3)  # 0=缺失, 1-3=有效类别
        mood_change_idx = torch.clamp(aux_attrs[:, 4], min=0, max=3)  # 0=缺失, 1-3=有效类别

        # Embedding查找
        family_emb = self.family_structure_embed(family_idx)  # (B, embed_dim)
        only_child_emb = self.only_child_embed(only_child_idx)
        parental_fav_emb = self.parental_favoritism_embed(parental_fav_idx)
        grade_change_emb = self.grade_change_embed(grade_change_idx)
        mood_change_emb = self.mood_change_embed(mood_change_idx)

        # 拼接所有embedding
        encoded = torch.cat([
            family_emb,
            only_child_emb,
            parental_fav_emb,
            grade_change_emb,
            mood_change_emb,
        ], dim=-1)  # (B, 5 * embed_dim)

        return self.dropout(encoded)
