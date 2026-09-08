# 模型模块文档索引

本目录包含 AdoDAS2026 Baseline 模型架构的详细讲解文档。

## 文档列表

1. [mtcn_backbone.md](mtcn_backbone.md) - 多模态TCN骨干网络详解
2. [heads.md](heads.md) - 任务预测头详解 (含 AuxAttributeHeads)
3. [grouped_model.md](grouped_model.md) - 参与者聚合模型详解 (含 LUPI 扩展)
4. [../lupi/](../lupi/) - LUPI 机制详解 (Phase 1/2)

## 模型架构总览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           模型架构总览                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  GroupedModel (参与者级模型)                                                 │
│  ├── MTCNBackbone (多模态TCN骨干)                                            │
│  │   ├── GroupAdapter (特征适配层)                                          │
│  │   ├── ModalityFusion (模态融合层)                                        │
│  │   ├── TCN (时序卷积网络)                                                  │
│  │   ├── ASP (注意力统计池化)                                                │
│  │   └── FusionMLP (最终融合)                                               │
│  ├── ParticipantAggregator (参与者聚合)                                      │
│  └── SessionTypeClassifier (会话类型分类)                                    │
│                                                                             │
│  TaskHead (任务预测头)                                                       │
│  ├── A1Head (三分类二元分类)                                                 │
│  ├── A2OrdinalHead (序数回归)                                                │
│  └── CORALHead (Consistent Rank Logits)                                     │
│                                                                             │
│  LUPI Extensions                                                             │
│  ├── AuxAttributeHeads (辅助属性预测头, 输入端)                               │
│  ├── AuxAttributeEncoder (辅助属性编码器, 输出端)                              │
│  └── _compute_aux_consistency_weight (样本加权, Phase 2)                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 核心设计原则

### 1. 多模态融合
- **早期融合**: 在模态内部融合多个特征组
- **中期融合**: 在骨干网络输出层融合音频和视频
- **注意力机制**: ASP池化结合VAD和QC信号

### 2. 时序建模
- **TCN架构**: 膨胀残差卷积，捕获多尺度时序模式
- **感受野控制**: 通过层数和膨胀率精确控制
- **并行计算**: 相比RNN更高效的训练

### 3. 参与者级聚合
- **多会话整合**: 每个参与者有4个会话
- **灵活聚合**: 支持mean/mlp/attention三种方法
- **多任务学习**: 主任务+会话级辅助任务

### 4. 序数回归
- **阈值分解**: 将序数问题转化为多个二元分类
- **单调性保证**: CORAL通过cumsum保证阈值单调
- **解码灵活**: 支持argmax/expectation/monotonic解码

## 模型参数统计

```
典型配置下的参数量 (d_model=256, d_adapter=64, d_shared=256):
┌─────────────────────────────────────────────────────────────────────────────┐
│  MTCNBackbone:                                                              │
│    - Adapters: ~10K per feature group                                      │
│    - Fusion layers: ~50K                                                   │
│    - TCN (6 layers): ~300K                                                 │
│    - ASP: ~1K                                                              │
│    - Final MLP: ~100K                                                      │
│    Total: ~500K - 1M                                                        │
│                                                                             │
│  GroupedModel:                                                              │
│    - Aggregator (MLP): ~130K                                               │
│    - Session classifier: ~16K                                              │
│                                                                             │
│  TaskHead:                                                                  │
│    - A1Head: ~800                                                          │
│    - A2OrdinalHead: ~16K                                                   │
│    - CORALHead: ~5K + ~60 thresholds                                        │
│                                                                             │
│  Grand Total: ~1M - 2M parameters                                          │
└─────────────────────────────────────────────────────────────────────────────┘
```