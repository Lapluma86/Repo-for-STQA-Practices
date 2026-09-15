# 软件测试与质量保证课程实践

以 SteelRailWay 钢轨缺陷检测项目中的数据加载、损失函数和工程指标为被测对象，开展测试基础实践与 AI 融合实践。

```text
SteelRailWay/                  被测系统源码及原有实验产物
tests/                         独立测试工程、设计文档和历史日志
deliverables/module1/           模块一正式格式交付件与执行证据
pytest.ini                     根目录测试入口及收集范围
.coveragerc                    指定被测模块的覆盖率配置
实践作业要求v2026.pdf           课程要求
实践作业-文档模板2026/          教师提供的交付模板
```

## 配置与运行

默认使用现有 Conda 环境 `AI_common`（Python 3.10）。在仓库根目录执行：

```powershell
Set-Location 'D:\projects\Repo-for-STQA-Practices'
& 'D:\Anaconda\envs\AI_common\python.exe' -m pip install --no-user -r tests/requirements-test.txt
& 'D:\Anaconda\envs\AI_common\python.exe' -m pytest
```

被测系统依赖的 PyTorch、torchvision、NumPy、SciPy、scikit-learn 和 Pillow 使用该环境的已有安装；首次准备环境应先确认可正常导入。
完整测试范围、分模块命令、覆盖率报告和课程交付清单见 [测试说明](tests/README.md)。
模块一为测试基础实践，包含自动化测试；模块二为 AI 融合实践。已有测试不代表正式交付件全部完成。
模块一现为50项测试，被测源码已回退并冻结，见[当前说明](tests/module1/REDUCTION_AND_FREEZE.md)。[交付包](deliverables/module1/README.md)中的Excel、Word、PPT及199项证据保留为历史版本。

## 协作

成员使用个人账号提交真实工作，提交信息使用中文并说明具体改动。报告中的成员贡献合计100%，应能对应 Git 历史。
保留缺陷修复前后证据与历史实验产物；真实数据和模型权重按源码目录的忽略规则管理。
