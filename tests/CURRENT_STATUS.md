# 当前测试状态（2026-09-15，精简及冻结后）

## 结果

| 范围 | 收集 | 通过 | 失败 | 跳过 | 执行错误 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 模块一 | 50 | 39 | 10 | 1 | 0 |
| 模块二（保留原用例） | 16 | 6 | 10 | 0 | 0 |
| 分别执行后的合计 | 66 | 45 | 20 | 1 | 0 |

使用 `D:\Anaconda\envs\AI_common\python.exe` / Python 3.10.20。
两模块分别执行，退出码均为1；模块一32.10秒，模块二60.01秒。不是同一次全量测试统计。
模块一有1项CUDA测试因设备不可用跳过。首次沙箱临时目录权限失败不计入上表，表内均来自沙箱外完整执行。

## 当前基线

- 模块一由199项按类别比例精简至50项：黑盒24、白盒25、集成1。三项指定缺陷各保留一个测试，全部复现失败。
- 三个被测文件回退至首次测试提交之前的 `e05aafbf8dcf320622b04109d3a8766229cea94f`，已核对字节完全一致。
- 86个产品Python文件记录在 [冻结指纹](source-freeze.json)；pytest开始时自动检查内容和缺失文件。原基线允许运行、模拟修改被拒绝均已验证。
- 模块一语句/分支综合覆盖70.78%，仅针对数据集、工程指标、损失函数三个模块，不是全项目覆盖率。
- 保留原断言，不使用xfail将缺陷隐藏。10个失败节点不等于10个独立缺陷。

模块一10个失败：三项指定缺陷、六项参数校验、一个真实时钟用例再次触发短时延为0。真实时钟结果可能随机器变化，另有可控时钟测试稳定复现同一缺陷。
模块二10个失败涉及patch归一化顺序/统计/预加载、参数拒绝和空列表返回类型。它们是回退后的影响记录，不自动认定为10个独立产品缺陷；例如空列表Tensor返回类型约定仍需需求审核。

[50项完整清单、比例及复现命令](module1/REDUCTION_AND_FREEZE.md)。

```powershell
& 'D:\Anaconda\envs\AI_common\python.exe' -m pytest -c pytest.ini tests/module1/
& 'D:\Anaconda\envs\AI_common\python.exe' -m pytest -c pytest.ini tests/module2/
```

## 执行证据与历史边界

本次证据：[模块一日志](logs/frozen-50/module1.log)、[模块一JUnit](logs/frozen-50/module1.xml)、[模块一HTML](logs/frozen-50/module1.html)、[模块二日志](logs/frozen-50/module2.log)、[模块二JUnit](logs/frozen-50/module2.xml)。
旧版当前状态保存为 [历史快照](logs/frozen-50/previous-current-status.md)。
旧Excel、Word、PPT、199/215项报告及修复后通过结论均属于历史版本；本次没有重新生成办公附件。
未进行训练、真实数据精度测试或CUDA实测；课程成员贡献及视频验收仍需小组处理。
