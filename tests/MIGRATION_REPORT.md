> 当前结论已更新：请参阅 [CURRENT_STATUS.md](CURRENT_STATUS.md) 和 [DEFECT_AUDIT.md](DEFECT_AUDIT.md)。以下保留历史原文。

# 测试工程迁移与验证记录

日期：2026-09-14。此记录对应本次目录迁移后的工作区，不替代课程正式测试报告。

## 变更

- `SteelRailWay/tests/` 独立为根目录 `tests/`。
- `manual/phase1/` 改为 `module1/`，`manual/phase2/` 改为 `module2/`；测试标记同步为 module1/module2。
- 删除各测试文件重复且不可靠的源码路径注入，由 conftest.py 按自身绝对位置统一导入 SteelRailWay。
- pytest.ini 移到根目录，默认只收集两个课程模块；覆盖率设置独立为 .coveragerc，包含语句和分支统计。
- 根目录 README 和 tests/README 按课程要求说明阶段、环境、命令、交付清单及现有缺口。
- 原始历史日志内容保持一致；旧 Markdown 增加历史说明。新执行产物写入 tests/artifacts/ 并本地忽略。
- 未修改被测系统业务代码或测试断言；本次验证保留发现的真实失败。

## 环境

Windows，AI_common / Python 3.10.20，PyTorch 2.12.0+cpu，torchvision 0.27.0+cpu，NumPy 1.23.5。
安装 pytest 9.1.1、pytest-cov 7.1.0、pytest-html 4.2.0、Hypothesis 6.168.0、OpenCV 4.11.0.86、tqdm 4.70.1 及其所需依赖。
依赖通过指定解释器 `-m pip install --no-user` 写入 AI_common。

## 最终验证

从仓库根目录执行：

```powershell
& 'D:\Anaconda\envs\AI_common\python.exe' -m pytest --cov --cov-config=.coveragerc --cov-report=term-missing --cov-report=html --html=tests/artifacts/final.html --self-contained-html --junitxml=tests/artifacts/final.xml
```

| 范围 | 收集 | 通过 | 失败 | 跳过 |
| --- | ---: | ---: | ---: | ---: |
| module1 | 172 | 157 | 4 | 11 |
| module2 | 7 | 6 | 1 | 0 |
| 合计 | 179 | 163 | 5 | 11 |

耗时30.17秒，退出码1。11个跳过项为 CUDA 测试，本环境无 CUDA。
首次沙箱运行受到 Windows 临时目录权限限制，最终验证在沙箱外使用同一解释器执行；不能将首次环境错误计为系统缺陷。

最终失败：

1. `test_very_long_path`：创建测试目录时遭遇 Windows 长路径限制，尚未调用被测构造函数。
2. `test_measure_inference_latency_cuda_false`、`test_measure_inference_latency_path_2`、`test_normal_path_vs_exception_path`：实测时延返回0，`latency > 0` 失败。被测实现使用 time.time() 测量很短的循环，失败项在两次运行间变化，存在时钟分辨率/计时稳定性问题，需单独修复与回归验证。
3. `test_config_combinations`：Hypothesis 默认200ms deadline 被预加载调用超过（记录319.62ms）。该用例验证参数组合，后续应明确功能测试超时策略与性能指标的区别。

最终日志：`tests/artifacts/final.log`；HTML：`tests/artifacts/final.html`；JUnit XML：`tests/artifacts/final.xml`。

开启 branch 后 coverage 报告的综合覆盖率为：数据集41.88%、工程指标88.00%、损失函数100.00%，三个指定模块合计53.88%。
这不是整个钢轨项目覆盖率，也不是模块一单独执行的覆盖率；不能与历史仅两个模块的语句覆盖率96%直接比较。

## 尚未完成的质量工作

本次范围为 README、结构迁移、依赖安装及执行验证。上述失败、既有数据加载测试可能空执行、缺陷清单的有效性审计，以及 Excel/Word/PPT/视频交付件仍需后续处理。
