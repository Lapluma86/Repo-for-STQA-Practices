# 测试工程

从仓库根目录运行，`conftest.py` 将 `SteelRailWay/` 加入导入路径。完整被测源码通过 `source-freeze.json` 在测试会话开始时校验；清理目录不改变冻结指纹。

| 路径 | 范围 |
| --- | --- |
| `module1/blackbox/` | 数据集等价类、边界值、判定表、状态与空间对齐 |
| `module1/whitebox/` | 损失函数、工程指标的语句/分支/条件/路径设计及数值断言 |
| `module1/test_pipeline.py` | 数据集→DataLoader→损失→反向传播集成验证 |
| `module2/ai_assisted/` | Hypothesis 参数组合、属性测试和缺陷验证 |

模块一48项（黑盒24、白盒23、集成1），模块二16项。用例数不等于 Hypothesis 生成示例数。

## 环境与运行

使用 `D:\Anaconda\envs\AI_common\python.exe`；测试工具及补充依赖见 `requirements-test.txt`，PyTorch、torchvision等运行依赖需在该环境中准备好。

```powershell
& 'D:\Anaconda\envs\AI_common\python.exe' -m pip install --no-user -r tests/requirements-test.txt
& 'D:\Anaconda\envs\AI_common\python.exe' -m pytest -c pytest.ini tests/module1/ tests/module2/
& 'D:\Anaconda\envs\AI_common\python.exe' -m pytest -c pytest.ini tests/ --collect-only -q
.\run_module1_tests.bat --no-pause
```

默认临时目录若因本机权限拒绝访问，可使用新的临时路径：

```powershell
$testTemp = Join-Path $env:TEMP ('stqa-tests-' + [guid]::NewGuid().ToString('N'))
& 'D:\Anaconda\envs\AI_common\python.exe' -m pytest -c pytest.ini tests/module1/ tests/module2/ --basetemp=$testTemp
```

`--basetemp` 会清理指定目录，必须使用专用的新路径。运行结果0表示全部通过，1表示测试失败，2及以上需查看具体错误。

## 覆盖率和报告

```powershell
New-Item -ItemType Directory -Force tests/artifacts | Out-Null
& 'D:\Anaconda\envs\AI_common\python.exe' -m pytest -c pytest.ini tests/module1/ --cov --cov-config=.coveragerc --cov-report=term-missing --cov-report=html --html=tests/artifacts/module1.html --self-contained-html --junitxml=tests/artifacts/module1.xml
```

覆盖率仅对应 `datasets.rail_dataset`、`utils.losses`、`eval.metrics_engineering`，不是全项目覆盖率，也不能代替条件/路径覆盖证明。生成的 `tests/artifacts/` 可清理；正式证据保存在 [docs/evidence](../docs/evidence/README.md)。

当前说明见 [测试文档索引](../docs/testing/README.md)。测试使用合成数据，无需真实钢轨数据或模型权重；当前CPU测试不验证完整TRD精度或GPU性能。
