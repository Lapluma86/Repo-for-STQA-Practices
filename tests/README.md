# 软件测试课程实践

测试工程位于仓库根目录 `tests/`，被测源码位于同级 `SteelRailWay/`。
从仓库根目录运行；`conftest.py` 统一设置被测源码的导入路径。

## 目录与测试范围

```text
tests/
├── conftest.py              # 共享 fixture、随机种子与源码导入
├── requirements-test.txt    # 测试工具及缺失运行依赖
├── module1/
│   ├── blackbox/            # 数据集参数、边界、组合与加载行为
│   ├── whitebox/            # 损失函数、工程指标
│   └── PHASE1_SUMMARY.md    # 历史阶段总结
├── module2/
│   └── ai_assisted/         # AI 辅助参数探索及回归测试
├── logs/                   # 原始历史执行证据，保留原文
└── artifacts/              # 本机新执行产物（Git 忽略）
```

模块一是“测试基础实践”，包括需求分析、用例设计和自动化执行；pytest 自动化属于本模块。
已有等价类、边界值、判定表和状态转换相关测试，以及白盒测试。
模块二是“AI 融合实践”，按课程要求在“测 AI”与“AI 测”中任选其一。
当前 `module2/ai_assisted/` 是“AI 测”的初步实现，不能据此认定模块二已经交付。
本项目按小组当前授权保留 AI 辅助成果，保留真实来源记录。

当前范围为 `datasets/rail_dataset.py`、`utils/losses.py`、`eval/metrics_engineering.py`。
不包含完整训练、真实数据上的模型精度验证或端到端检测验收；这些需要另行准备数据、权重和资源。
测试生成的图像位于 pytest 临时目录；`SteelRailWay/datasets/` 是源码，不是随仓库分发的数据集。

## 环境与一键运行

使用 Windows Conda 环境 `AI_common`（Python 3.10），先确保其中的 PyTorch、torchvision、NumPy、SciPy、scikit-learn、Pillow 能正常导入。
已有 PyTorch 为 CPU 版本时，CUDA 测试会跳过；跳过不代表验证通过。

```powershell
Set-Location 'D:\projects\Repo-for-STQA-Practices'
& 'D:\Anaconda\envs\AI_common\python.exe' -m pip install --no-user -r tests/requirements-test.txt

# 一键执行全部用例；根目录 pytest.ini 限定收集范围
& 'D:\Anaconda\envs\AI_common\python.exe' -m pytest

# 分模块执行
& 'D:\Anaconda\envs\AI_common\python.exe' -m pytest tests/module1/
& 'D:\Anaconda\envs\AI_common\python.exe' -m pytest tests/module2/

# 模块一报告与指定三个模块的语句/分支覆盖率
New-Item -ItemType Directory -Force tests/artifacts | Out-Null
& 'D:\Anaconda\envs\AI_common\python.exe' -m pytest tests/module1/ --cov --cov-config=.coveragerc --cov-report=term-missing --cov-report=html --html=tests/artifacts/module1.html --self-contained-html --junitxml=tests/artifacts/module1.xml
```

统计应以同一提交的一次执行为准，分别列出通过、失败、跳过；覆盖率仅代表明确列出的被测模块。
语句/分支覆盖率不能代替条件覆盖或路径覆盖的证明。

## 课程交付要求

课程要求和报告模板位于仓库根目录；按文档的其余要求准备成果：

| 项目 | 模块一 | 模块二 |
| --- | --- | --- |
| Excel 用例清单 | ≥30条；等价类/边界值/场景法中至少2种 | AI相关或AI生成用例≥15条 |
| 自动化源代码 | 一键运行并附环境说明 | 一键运行并附环境说明 |
| Word 缺陷清单 | 有效缺陷≥3个，含复现和修复验证 | 有效缺陷≥1个 |
| Word 测试报告 | 对象、范围、设计、实现、结果、缺陷、贡献 | 另需 AI 方案、实践过程和反思 |
| 汇报 | PPT及演示视频 | PPT及演示视频 |

模块一报告模板还要求自动化比例≥80%，分析5–8个典型用例、2–3个典型缺陷，并建立需求到用例的映射。
成员贡献合计100%，与各成员个人账号的真实 Git 提交对应；被测代码、脚本、数据生成逻辑、配置和关键 AI 对话应纳入版本管理。

现有 Markdown 清单、总结和缺陷报告属于历史素材，存在过期数字及待核实结论，不等同于填写完成的 Excel/Word/PPT/视频。
`logs/` 的历史路径和旧阶段名称为原始记录；迁移前 `manual/phase1/` 对应现在的 `module1/`，`manual/phase2/` 对应 `module2/`。
当前目录未包含上述全部正式交付件，不能声明课程作业已全部完成。
历史模块一交付材料保留在[交付包](../deliverables/module1/README.md)，其中199项执行证据对应精简前、源码修复后版本。

## 当前执行结果

模块一已按比例从199项精简为50项；模块二保留16项，全部测试共66项。
被测源码已回退并冻结，测试会如实暴露缺陷，退出码非零并不等于执行未完成。
当前结果、50项用例清单、冻结范围及复现命令见 [当前状态](CURRENT_STATUS.md) 和 [精简与冻结说明](module1/REDUCTION_AND_FREEZE.md)。
旧版Excel、Word、PPT及199/215项执行报告属于修复后历史版本，不代表当前冻结版本。
