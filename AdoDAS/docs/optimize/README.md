# ADODAS 全局优化策略

本目录包含针对 ADODAS 比赛的系统性优化方案，覆盖数据、特征、模型、损失函数和训练策略五个层面。

## 目录结构

```
optimize/
├── README.md                           # 本文件：全局优化策略总览
├── 01_data_quality/                    # 数据质量优化
│   ├── adaptive_mask.py               # 自适应掩码策略
│   ├── interpolated_alignment.py      # 混合时间对齐策略
│   └── README.md
├── 02_data_augmentation/              # 数据增强优化
│   ├── temporal_augmentation.py       # 时序数据增强（Time Masking + Speed Perturb）
│   ├── modality_dropout.py            # 模态整体 dropout
│   └── README.md
├── 03_feature_engineering/            # 特征工程优化
│   ├── feature_normalizer.py          # 特征 z-score 归一化
│   ├── vad_enhancement.py             # VAD 信号增强
│   └── README.md
├── 04_utils/                          # 工具函数优化
│   ├── metrics_optimized.py           # F1 阈值搜索 + QWK 向量化
│   ├── ckpt_optimized.py              # 增强检查点 + Top-K 管理
│   └── README.md
├── 05_model_architecture/             # 模型架构优化 ★ 新增
│   ├── cross_modal_attention.py       # 跨模态交叉注意力
│   ├── modality_gating.py             # 自适应模态门控融合
│   ├── aggregator_enhanced.py         # 增强的跨会话聚合器
│   └── README.md
├── 06_loss_functions/                 # 损失函数优化 ★ 新增
│   ├── asymmetric_loss.py             # ASL + Soft-F1（A1 任务）
│   ├── ordinal_loss_enhanced.py       # CORN + 可微 QWK（A2 任务）
│   └── README.md
└── 07_training_strategy/              # 训练策略优化 ★ 新增
    ├── mixup.py                       # Mixup 数据增强
    ├── multi_sample_dropout.py        # 多采样 Dropout
    ├── swa.py                         # 随机权重平均 (SWA)
    ├── ema.py                         # 指数移动平均 (EMA)
    └── README.md
```

## 发现的关键问题

### 代码缺陷（必须修复）

| 问题 | 位置 | 影响 | 修复方案 |
|------|------|------|----------|
| `a1_loss` 签名不匹配 | heads.py vs runner.py | **运行时报错** | `06_loss_functions/asymmetric_loss.py` |
| `a2_ordinal_loss` 签名不匹配 | heads.py vs runner.py | **运行时报错** | `06_loss_functions/ordinal_loss_enhanced.py` |
| ASL/CORN/QWK 损失未实现 | heads.py | 配置声明但无效果 | 同上 |
| BackboneConfig 缺少层次化编码器参数 | mtcn_backbone.py | 配置传入被静默忽略 | 需评估是否真正需要 |
| submissions 目录未创建 | run_naming.py | 推理时 FileNotFoundError | runner.py 已有 workaround |

### 架构瓶颈（提升空间最大）

| 瓶颈 | 当前做法 | 问题 | 优化方案 |
|------|----------|------|----------|
| 跨模态融合过晚 | 最终层线性拼接 | 无法捕捉时序级跨模态关联 | `05_model_architecture/cross_modal_attention.py` |
| 模态权重固定 | 拼接后线性投影 | 低质量模态干扰预测 | `05_model_architecture/modality_gating.py` |
| 训练正则化不足 | 仅 feature_noise + session_drop | ~1000 样本严重过拟合 | `07_training_strategy/mixup.py` |
| 单 checkpoint 方差大 | 保存最佳 epoch | 不同 seed 差 5%+ | `07_training_strategy/swa.py` + `ema.py` |

## 优化策略优先级总表

### P0（必做，预期综合提升 10~20%）

| 序号 | 优化项 | 目录 | 改动量 | 预期提升 |
|------|--------|------|--------|----------|
| 1 | **修复损失函数接口** | 06_loss_functions | 替换2个函数 | 修复运行时错误 |
| 2 | **ASL + Soft-F1 联合损失** | 06_loss_functions | 同上 | A1 F1 +3~8% |
| 3 | **CORN + 可微 QWK 损失** | 06_loss_functions | 同上 | A2 QWK +3~8% |
| 4 | **Mixup 正则化** | 07_training_strategy | 训练循环加3行 | QWK/F1 +2~5% |
| 5 | **特征归一化** | 03_feature_engineering | 数据集加一步 | 收敛加速 +3~5% |
| 6 | **时序数据增强** | 02_data_augmentation | 数据集集成 | QWK +5~10% |
| 7 | **F1/QWK 阈值搜索** | 04_utils | 已实现，需集成 | F1/QWK +2~5% |

### P1（推荐，预期额外提升 5~10%）

| 序号 | 优化项 | 目录 | 改动量 | 预期提升 |
|------|--------|------|--------|----------|
| 8 | 跨模态交叉注意力 | 05_model_architecture | backbone 加一层 | QWK +2~5% |
| 9 | 自适应模态门控 | 05_model_architecture | 替换 fusion_mlp | QWK +1~3% |
| 10 | 增强跨会话聚合器 | 05_model_architecture | 替换 aggregator | QWK +1~3% |
| 11 | Multi-Sample Dropout | 07_training_strategy | 包装 task_head | QWK/F1 +1~2% |
| 12 | EMA / SWA | 07_training_strategy | 训练循环加几行 | QWK +1~3% |
| 13 | 自适应掩码 | 01_data_quality | 替换掩码方法 | 数据利用率 +20% |
| 14 | 模态 Dropout | 02_data_augmentation | 训练循环加一步 | 鲁棒性 +15% |
| 15 | Top-K 检查点 | 04_utils | 替换保存逻辑 | 防止训练事故 |

### P2（可选，边际收益）

| 序号 | 优化项 | 目录 | 预期提升 |
|------|--------|------|----------|
| 16 | 混合时间对齐 | 01_data_quality | 2~3% |
| 17 | VAD 特征增强 | 03_feature_engineering | 2~3% |

## 快速集成路径

### 最小改动方案（修复错误 + 损失函数升级，30分钟）

```python
# 步骤1：将 asymmetric_loss.py 中的 a1_loss_enhanced 复制到 heads.py
# 步骤2：将 ordinal_loss_enhanced.py 中的 a2_ordinal_loss_enhanced 复制到 heads.py
# 步骤3：更新导入

# 在 heads.py 末尾：
# 替换原 a1_loss → a1_loss_enhanced（函数名改为 a1_loss 即可）
# 替换原 a2_ordinal_loss → a2_ordinal_loss_enhanced（同理）
```

### 标准方案（P0 全部实施，1~2天）

在最小改动基础上：
1. 集成特征归一化（`03_feature_engineering/feature_normalizer.py`）
2. 集成时序增强（`02_data_augmentation/temporal_augmentation.py`）
3. 训练循环加入 Mixup（`07_training_strategy/mixup.py`）

### 完整方案（P0+P1，3~5天）

在标准方案基础上：
1. 加入跨模态注意力和门控融合
2. 替换聚合器
3. 加入 EMA + SWA
4. 加入 Multi-Sample Dropout

## 各层优化的独立性

```
数据层      ──→ 特征层      ──→ 模型层       ──→ 损失层       ──→ 训练层
01/02/03          03            05              06              07/04

每层优化独立有效，可以按优先级逐步叠加。
低层优化（数据/特征）改善输入质量，高层优化（模型/损失/训练）改善学习效率。
同时优化多层通常有叠加效果（非线性组合，总提升 > 各项之和）。
```

## 参考文献

- Asymmetric Loss: Ridnik et al., ICCV 2021
- CORN: Shi et al., Pattern Recognition 2021
- Mixup: Zhang et al., ICLR 2018
- Multi-Sample Dropout: Inoue, 2019
- SWA: Izmailov et al., UAI 2018
- Bottleneck Transformers: Nagrani et al., ICML 2021
- SpecAugment: Park et al., Interspeech 2019
