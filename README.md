# 软件测试与质量保证课程实践

以 SteelRailWay 钢轨缺陷检测项目为被测系统，开展模块一测试基础实践与模块二 AI 融合实践。完整保留被测系统源码、训练/评估脚本和配置；当前自动化测试针对数据加载、损失函数与工程指标三个模块。

| 目录 | 内容 |
| --- | --- |
| `SteelRailWay/` | 完整被测源码、配套配置及脚本 |
| `tests/` | 模块一、模块二测试代码与运行配置 |
| [docs/testing](docs/testing/README.md) | 当前用例设计、缺陷、数据说明及测试附件 |
| [docs/evidence](docs/evidence/README.md) | 当前测试证据，按执行批次注明口径 |
| [docs/course](docs/course/README.md) | 教师原始要求与模板 |

## 运行测试

默认使用 Conda 环境 `AI_common`。在仓库根目录执行：

```powershell
& 'D:\Anaconda\envs\AI_common\python.exe' -m pytest -c pytest.ini tests/module1/ tests/module2/
```

双击 `run_module1_tests.bat` 可运行模块一；终端中可加 `--no-pause`。详细依赖、报告命令和临时目录权限处理见 [测试 README](tests/README.md)。

模块一保留48项CPU测试，模块二保留16项；被测源码冻结，已有缺陷会造成测试失败。当前状态见 [CURRENT_STATUS.md](docs/testing/CURRENT_STATUS.md)。测试失败不等于运行未完成，也不能通过修改断言或源码指纹掩盖。

真实数据、预训练权重和模型 checkpoint 不随仓库提供。完整训练和真实数据评估需自行准备资源；训练入口从 `SteelRailWay/` 运行，参数以对应 argparse 为准。

## 文件管理

研究输出、缓存和新生成的测试报告不纳入版本控制。必要执行证据集中到 `docs/evidence/`，避免重复副本。汇报PPT、演示材料与过期文档已移除。Git提交信息使用中文，本地 `AGENTS.md` 和 `.vscode/` 不提交。
