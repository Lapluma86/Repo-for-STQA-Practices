# AdoDAS2026 Baseline 项目文档

## 概述

AdoDAS2026 Baseline 是为 ACM MM 2026 AdoDAS 大赛开发的官方基线实现。该项目旨在通过多模态深度学习模型识别青少年的抑郁、焦虑、压力等心理健康问题。

## 项目架构

```
AdoDAS2026-main/
├── common/                     # 核心模块包
│   ├── data/                   # 数据处理模块
│   │   ├── dataset.py          # 单会话数据集处理
│   │   ├── feature_io.py       # 特征文件I/O操作
│   │   └── grouped_dataset.py  # 多会话分组数据集处理
│   ├── models/                 # 模型定义模块
│   │   ├── mtcn_backbone.py    # 多模态TCN骨干网络
│   │   ├── heads.py            # 任务预测头
│   │   └── grouped_model.py    # 分组参与者模型
│   ├── utils/                  # 工具函数模块
│   │   ├── ckpt.py             # 检查点保存/加载
│   │   ├── metrics.py          # 评估指标计算
│   │   ├── seed.py             # 随机种子设置
│   │   ├── run_naming.py       # 运行目录命名
│   │   └── run_metadata.py     # 运行元数据管理
│   └── runner.py               # 训练和验证主逻辑
├── tasks/                      # 任务配置文件
│   ├── a1/default.yaml         # A1任务配置
│   └── a2/default.yaml         # A2任务配置
├── envs/                       # 环境配置
│   └── adodas.yaml             # Conda环境配置
├── train.py                    # 训练脚本入口
├── infer.py                    # 推理脚本入口
└── docs/                       # 项目文档
    ├── architecture.md         # 架构总览
    ├── data/                   # 数据模块文档
    ├── models/                 # 模型模块文档
    ├── utils/                  # 工具模块文档
    └── guides/                 # 使用指南
```

## 任务说明

本项目包含两个主要任务：

### Track A1 (三分类任务)
- **目标**: 识别抑郁(Depression)、焦虑(Anxiety)、压力(Stress)三种心理状态
- **输出**: 3个二元标签概率值 (p_D, p_A, p_S)
- **损失函数**: 带权重的二元交叉熵损失
- **评估指标**: Macro F1-score, AUROC

### Track A2 (序数回归任务)
- **目标**: 预测21个心理评估项目的分数
- **输出**: 21个整数分数，取值范围 {0, 1, 2, 3}
- **方法**: 序数回归(Ordinal Regression) / CORAL (Consistent Rank Logits)
- **评估指标**: Mean Quadratic Weighted Kappa (QWK), Mean Absolute Error (MAE)

## 技术架构

### 1. 数据流

```
原始特征文件 (.npz, .parquet)
    ↓
Feature I/O (feature_io.py)
    ↓
数据对齐 (align_to_grid)
    ↓
数据集 (GroupedParticipantDataset)
    ↓
数据批次 (grouped_collate_fn)
    ↓
MTCNBackbone (多模态TCN骨干网络)
    ↓
GroupedModel (参与者聚合)
    ↓
Task Head (A1Head / A2OrdinalHead / CORALHead)
    ↓
预测输出
```

### 2. 模型架构

```
输入层 (多模态特征组)
    ├── 音频序列特征 (mel_mfcc, vad, ssl_embed)
    ├── 音频池化特征 (egemaps)
    └── 视频特征 (headpose_geom, face_behavior, vision_ssl_embed, ...)
         ↓
特征适配层 (GroupAdapter)
    └── 将不同维度的特征投影到统一维度 d_adapter
         ↓
模态融合层 (ModalityFusion)
    ├── 音频模态融合
    └── 视频模态融合
         ↓
时序建模层 (TCN)
    ├── 音频TCN (Dilated Residual Blocks)
    └── 视频TCN
         ↓
注意力统计池化 (ASP - Attentive Statistics Pooling)
    └── 结合VAD信号和质量控制信号生成统计特征
         ↓
多会话聚合 (ParticipantAggregator)
    └── 聚合4个会话(A01, B01, B02, B03)的特征
         ↓
任务预测头
    ├── A1: 3类二元分类
    └── A2: 21项序数回归
```

## 核心特性

### 1. 多模态融合
- **音频特征**: MFCC、Mel频谱、VAD、SSL嵌入、egemaps
- **视频特征**: 头部姿态、面部行为、身体姿态、全局运动、SSL嵌入
- **融合策略**: 早期融合(模态内) + 中期融合(模态间) + 注意力池化

### 2. 时序建模
- **TCN架构**: 膨胀残差卷积网络，捕获多尺度时序模式
- **膨胀率**: 指数增长 (1, 2, 4, 8, ...)
- **感受野**: 随层数指数增长，捕获长期依赖

### 3. 注意力机制
- **ASP池化**: 结合VAD(语音活动检测)和质量控制信号
- **会话聚合**: 支持mean/mlp/attention三种聚合策略

### 4. 训练技巧
- **类别不平衡处理**: 正样本权重计算
- **标签平滑**: 缓解过拟合
- **特征噪声**: 数据增强
- **会话丢弃**: 增强泛化能力
- **早停机制**: 防止过拟合
- **学习率调度**: Warmup + Cosine Annealing

## 5. LUPI 机制（Learning Using Privileged Information）

利用训练时可用但测试时不可用的辅助属性（家庭结构、独生子女等 5 个维度）：

- **Aux Heads — 辅助属性预测头**：从 participant_repr 预测辅助属性，迫使 backbone 学习相关潜变量
- **Sample Reweight — 样本一致性加权**：基于 aux_emotional 与 DASS 标签一致性识别错标样本

详见 `docs/AUX_LUPI_PLAN.md` 和 `docs/lupi/`。

## 文档索引

- [架构总览](architecture.md) - 系统架构和模块关系
- [数据模块](data/) - 数据处理和加载详解
- [模型模块](models/) - 模型架构详解
- [工具模块](utils/) - 工具函数详解
- [LUPI 机制](lupi/) - 特权信息学习实现
- [使用指南](guides/) - 训练和推理指南

## 快速开始

### 环境安装
```bash
conda env create -f envs/adodas.yaml
conda activate adodas
```

### 启动脚本（推荐）
```bash
# A2 基线训练
./run_train.sh --task a2 --preset default

# A2 全量优化（MTL + 增强损失，LUPI 关闭）
./run_train.sh --task a2 --preset full

# A2 全量 + LUPI aux_heads
./run_train.sh --task a2 --preset full --lupi heads

# A2 全量 + LUPI 全部
./run_train.sh --task a2 --preset full --lupi both

# 快速调试
./run_train.sh --task a2 --preset debug
```
A2 各种训练模式
```bash
# 基线（什么都不开）
./run_train.sh --task a2 --preset default

# 仅 LUPI aux_heads（辅助属性预测头）
./run_train.sh --task a2 --preset default --lupi heads

# 仅 LUPI sample_reweight（样本一致性加权）
./run_train.sh --task a2 --preset default --lupi weight

# 两者都启用
./run_train.sh --task a2 --preset default --lupi both

# 全量 MTL 预设（LUPI 全部关闭）
./run_train.sh --task a2 --preset full

# 全量 MTL + LUPI aux_heads
./run_train.sh --task a2 --preset full --lupi heads

# 全量 MTL + LUPI sample_reweight
./run_train.sh --task a2 --preset full --lupi weight

# 全量 MTL + LUPI 全部
./run_train.sh --task a2 --preset full --lupi both
```

数据加载模式
```bash
# HDF5 全量预载（推荐：稳定、无 multiprocessing 死锁风险）
./run_train.sh --task a2 --preset full --gpu 0 --extra "--use_hdf5 1 --preload 1"

# HDF5 按需加载（内存更低，训练时 mmap 读盘）
./run_train.sh --task a2 --preset full --gpu 0 --extra "--use_hdf5 1 --preload 0"

# Raw 文件预载（需 preload_workers 控制并行数，过高易死锁）
./run_train.sh --task a2 --preset full --gpu 0 --extra "--preload 1 --preload_workers 4"
```

| preset | 配置 | 用途 |
|--------|------|------|
| `default` | `tasks/{task}/default.yaml` | 基线训练 |
| `full` | `tasks/{task}/mtl_full.yaml` | 全量优化 (MTL + 增强损失, LUPI 默认关闭) |
| `debug` | default + 覆盖 | 100 人, 流式加载, 2 epochs 秒级启动 |

### 直接使用 train.py
```bash
python train.py --task a2 --config tasks/a2/default.yaml
python train.py --task a2 --config tasks/a2/default.yaml \
    --aux_lupi_enabled 1 --aux_lupi_heads 1 --aux_lupi_reweight 1
```

### 推理
```bash
python infer.py --task a2 --checkpoint <path_to_best.pt> --split test_hidden
```
