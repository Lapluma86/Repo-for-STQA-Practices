# LUPI 机制文档（Learning Using Privileged Information）

## 概述

辅助属性（家庭结构、独生子女、父母偏爱、成绩变动、情绪变动）仅在训练集存在，测试时不可用。这是经典的 **LUPI** 范式。

所有改动满足以下硬约束：
1. 推理时不依赖辅助属性
2. `aux_lupi.enabled: false` 时完全回退到 baseline 行为
3. Phase 1/2 可独立启用

## 配置

```yaml
aux_lupi:
  enabled: true              # 总开关
  phase1_mtl:                # Phase 1: 辅助属性多任务监督
    enabled: true
    hidden: 64
    weights:
      aux_family: 0.05
      aux_only_child: 0.05
      aux_favoritism: 0.05
      aux_academic: 0.15
      aux_emotional: 0.20
  phase2_reweight:           # Phase 2: 样本一致性加权
    enabled: false
    weight_low: 0.7
    weight_high: 1.2
```

CLI 覆盖：
```bash
--aux_lupi_enabled 1 --aux_lupi_phase1 1 --aux_lupi_phase2 1
```

启动脚本：
```bash
./run_train.sh --task a2 --preset default --lupi p1+p2
./run_train.sh --task a2 --preset phase1 --lupi p2
```

## Phase 1: 多任务辅助监督

**原理**: 从 participant_repr (纯音视频表示) 预测 5 个辅助属性，迫使 backbone 编码与心理状态相关的潜变量。

**实现**: `common/models/heads.py: AuxAttributeHeads`

```
participant_repr (B, d_shared)
    │
    ├──→ AuxAttributeHeads → aux_logits (5 个分类头)
    │       └──→ aux_attribute_loss (CE, 缺失值 mask)
    │
    └──→ concat(aux_encoded) → 任务头 → 主任务预测
```

**关键**: AuxAttributeHeads 作用于 aux_encoder 拼接**之前**，确保 backbone 在无辅助属性时也学到相关表示。

**缺失值处理**: `aux_favoritism` 有 ~35% 结构性缺失 (独生子女无此属性)。损失函数通过 `valid = aux_attrs[:, i] >= 0` 自动跳过。

**权重推荐**: `aux_emotional`=0.20, `aux_academic`=0.15, 其他=0.05。总辅助损失约为任务的 1/3~1/2。

## Phase 2: 样本一致性加权

**原理**: `aux_emotional` (情绪变动) 与 DASS 标签之间存在天然一致性。利用此识别可能错标的样本并降权。

**实现**: `common/runner.py: _compute_aux_consistency_weight`

| 标签 | 情绪变动 | 判断 | 权重 |
|------|---------|------|------|
| DASS 阳性 | 变差 (3) | 一致 | 1.2 (加权) |
| DASS 阴性 | 变好 (1) | 一致 | 1.2 (加权) |
| DASS 阳性 | 变好 (1) | 冲突 | 0.7 (降权, 可能错标) |
| DASS 阴性 | 变差 (3) | 冲突 | 0.7 (降权, 可能错标) |
| 任意 | 无变化/缺失 | 中性 | 1.0 |

**非 MTL 模式**: 加权 per-sample BCE 作为附加项 (系数 0.3)，不覆盖增强损失。
**MTL 模式**: 加权 BCE 附加到 compute_optimized_loss 的结果上。

## 文件索引

| 文件 | 相关内容 |
|------|---------|
| `common/runner.py` | 训练循环中 aux_loss 合成, 样本权重计算, CLI 参数 |
| `common/models/heads.py` | `AuxAttributeHeads`, `aux_attribute_loss` |
| `common/models/grouped_model.py` | `GroupedModel.aux_heads`, `forward` 中 aux_logits |
| `common/models/phase1_integration.py` | MTL 模式 aux_logits 透传 |
| `common/utils/ckpt.py` | `load_checkpoint(strict=False)` |
| `infer.py` | `strict=False` 加载, `participant_head_state_dict` |
| `tasks/{a1,a2}/default.yaml` | `aux_lupi` 配置块 |
| `tasks/{a1,a2}/phase1_optimization.yaml` | `aux_lupi` 配置块 (Phase1 默认启用) |
| `run_train.sh` | `--lupi p1/p2/p1+p2` 启动脚本 |

## 调试

- **辅助 loss 不下降**: 检查 `aux_attrs` 是否正确加载 (打印 batch 中的值); 检查 `valid_mask` 是否处理 -1
- **主任务变差**: 辅助权重过高。整体除以 2 重试
- **训练发散**: 检查辅助头初始化; 检查梯度范数
- **关闭后 baseline 行为变了**: 检查 `enabled=false` 分支是否意外执行
