# 模型架构优化

## 问题诊断

通过分析 `common/models/` 下的代码和 `runner.py` 的训练逻辑，发现以下架构层面的瓶颈：

### 1. 跨模态融合过晚（最大瓶颈）

当前架构中，音频和视频通路在骨干网络中完全独立处理，直到最后一步才用线性拼接融合。这意味着模型无法捕捉时序级别的跨模态关联（如"说话时面部表情的变化"、"语速加快时同步的肢体动作"）。

**影响**: 丢失了多模态数据最有价值的信息——模态间的时序协同。

### 2. 模态融合权重固定（medium）

`fusion_mlp` 对 `z_audio` 和 `z_video` 的拼接做线性投影，隐含地赋予两个模态固定的权重。但实际中不同样本的模态质量差异巨大（录音故障、面部遮挡等），需要自适应的模态权重。

**影响**: 低质量模态的噪声会拖累整体预测。

### 3. 跨会话聚合过于简单（medium）

当前注意力聚合器仅用 `Linear(d_in, 1)` 计算会话权重，不区分会话类型（A01 自由对话 vs B01-B03 结构化任务），也不建模会话间的关系。

**影响**: 未利用会话类型的先验知识，也无法识别离群会话。

## 改进方案

| 文件 | 方案 | 预期提升 | 优先级 | 侵入性 |
|------|------|----------|--------|--------|
| `cross_modal_attention.py` | TCN后插入跨模态交叉注意力 | QWK +2~5% | P0 | 中 |
| `modality_gating.py` | 自适应门控替代简单拼接 | QWK +1~3% | P1 | 低 |
| `aggregator_enhanced.py` | 条件/自注意力聚合器 | QWK +1~3% | P1 | 低 |

## 文件说明

### cross_modal_attention.py

- **CrossModalAttention**: 单向跨模态注意力，Q 来自模态 A，KV 来自模态 B
- **BidirectionalCrossModalFusion**: 双向版本，音频和视频互相关注
- 插入位置：TCN（Step 3）和 ASP（Step 4）之间
- 设计决策：使用单头注意力（数据量小），加残差连接（安全）

### modality_gating.py

- **ModalityGating**: σ(W·[z_a; z_v]) 门控，每个特征维度独立权重
- **GatedFusionMLP**: 完整替换模块，门控融合 + MLP
- 替换目标：`MTCNBackbone.fusion_mlp`
- 额外优势：减少 fusion_mlp 输入维度（降低过拟合风险）

### aggregator_enhanced.py

- **ConditionalAttentionAggregator**: 引入会话类型嵌入作为注意力条件
- **SelfAttentionAggregator**: 多头自注意力建模会话间关系
- 替换目标：`GroupedModel.aggregator`

## 集成顺序建议

1. **先试 modality_gating**（侵入性最低，替换一个模块即可）
2. **再试 cross_modal_attention**（需要修改 backbone forward 方法）
3. **最后试 aggregator_enhanced**（替换 aggregator，需要传入 session_types）

每个改进独立有效，可以叠加使用。
