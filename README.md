# Repo-for-STQA-Practices

软件测试与质量保证实践仓库。当前模块一被测对象为 **SteelRailWay**（钢轨 RGB+Depth 缺陷检测）。

## 仓库结构

```
.
├── README.md                          # 本文件：结构、环境、如何跑测试
├── docs/module1/                      # 模块一正式交付文档
│   ├── 测试用例清单.xlsx
│   ├── 缺陷清单.docx
│   └── 测试报告.docx
├── 实践作业-文档模板2026/               # 课程模板
├── 实践作业要求v2026.pdf
├── SteelRailWay/                      # 被测系统 + 测试脚本
    ├── datasets/rail_dataset.py
    ├── utils/losses.py
    ├── eval/metrics_engineering.py
    └── tests/                         # pytest 工程
                          # 仓库内另一项目，非本阶段被测对象
```

被测模块：

| 模块 | 路径 | 本阶段测什么 |
|------|------|----------------|
| 双模态数据集 | `SteelRailWay/datasets/rail_dataset.py` | 参数校验、组合配置、对象状态 |
| 蒸馏损失 | `SteelRailWay/utils/losses.py` | 语句/分支/条件/路径覆盖 |
| 工程指标 | `SteelRailWay/eval/metrics_engineering.py` | 参数量、时延、显存、误报率 |

不需要真实钢轨图像或模型权重。黑盒用 `tmp_path` 建空目录，白盒用 `torch.randn` 与 Dummy 模型。

## 环境配置

- Python 3.8+
- PyTorch 2.x（CPU 即可；有 NVIDIA GPU 时可跑 11 条 CUDA 用例）
- pytest ≥ 7.4

```bash
cd SteelRailWay
pip install -r tests/requirements-test.txt
```

若尚未安装 PyTorch，请先按官方说明安装，再装上述测试依赖。

## 运行测试

在 `SteelRailWay/` 目录下：

```bash
# 模块一全部用例（推荐）
pytest tests/manual/phase1/ -v

# 仅黑盒
pytest tests/manual/phase1/blackbox/ -v

# 仅白盒
pytest tests/manual/phase1/whitebox/ -v

# 覆盖率
pytest tests/manual/phase1/whitebox/ \
    --cov=utils.losses \
    --cov=eval.metrics_engineering \
    --cov-report=html:tests/manual/phase1/coverage_reports \
    --cov-report=term-missing -v

# 无 GPU 时跳过 CUDA 用例
pytest tests/manual/phase1/ -v -k "not cuda"
```

历史执行记录见 `SteelRailWay/tests/logs/`（2026-09-13：172 条中 161 通过、11 条无 CUDA 跳过、0 失败；覆盖率约 96%）。

`tests/manual/phase2/` 为后续 AI 辅助目录，**不纳入模块一交付**。

## 模块一文档

| 文档 | 说明 |
|------|------|
| [docs/module1/测试用例清单.xlsx](docs/module1/测试用例清单.xlsx) | 172 条用例（等价类/边界值/判定表/状态转换 + 四类覆盖） |
| [docs/module1/缺陷清单.docx](docs/module1/缺陷清单.docx) | 7 个缺陷（现象、复现、修复验证） |
| [docs/module1/测试报告.docx](docs/module1/测试报告.docx) | 被测对象、策略、结果与贡献度（贡献度待补） |
