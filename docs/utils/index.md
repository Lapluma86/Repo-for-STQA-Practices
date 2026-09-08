# 工具模块文档索引

本目录包含 AdoDAS2026 Baseline 工具函数的详细讲解文档。

## 文档列表

1. [metrics.md](metrics.md) - 评估指标计算详解
2. [ckpt.md](ckpt.md) - 检查点保存/加载详解
3. [utils_other.md](utils_other.md) - 其他工具函数详解

## 模块功能概览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            工具模块功能概览                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  metrics.py (评估指标)                                                       │
│  ├── binary_f1()        - 二元F1分数                                        │
│  ├── per_class_f1()     - 每类F1分数                                        │
│  ├── macro_auroc()      - 宏平均AUROC                                       │
│  ├── mean_qwk()         - 平均QWK                                           │
│  ├── per_item_qwk()     - 每项QWK                                           │
│  └── mean_mae()         - 平均MAE                                           │
│                                                                             │
│  ckpt.py (检查点管理)                                                        │
│  ├── save_checkpoint()  - 保存模型检查点                                    │
│  └── load_checkpoint()  - 加载模型检查点                                    │
│                                                                             │
│  seed.py (随机种子)                                                          │
│  └── seed_everything()  - 设置全局随机种子                                  │
│                                                                             │
│  run_naming.py (运行命名)                                                    │
│  ├── build_run_name()   - 构建运行名称                                      │
│  └── setup_run_dirs()   - 创建运行目录结构                                  │
│                                                                             │
│  run_metadata.py (元数据管理)                                                │
│  ├── RunMetadata       - 运行元数据类                                       │
│  ├── update_best()     - 更新最佳指标                                       │
│  ├── finish()          - 标记运行完成                                      │
│  └── set_extra()       - 设置额外字段                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 指标使用场景

| 任务 | 主要指标 | 次要指标 |
|------|----------|----------|
| A1 | Macro F1 | AUROC, Per-class F1 |
| A2 | Mean QWK | Mean MAE, Per-item QWK |

## 检查点内容

```
检查点文件 (best.pt) 内容:
├── epoch: int               # 当前epoch
├── model_state_dict: dict   # 模型参数
├── optimizer_state_dict: dict  # 优化器状态
├── best_metric: float       # 最佳指标值
└── head_state_dict: dict    # 任务头参数
```