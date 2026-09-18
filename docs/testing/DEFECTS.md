# 当前缺陷记录

被测源码处于冻结状态，下列三个重点缺陷当前未修复。测试预期保留，失败用于暴露实际行为。输入、预期、实际结果以对应测试及执行证据相互核对。

| 编号 | 位置与触发条件 | 预期 | 实际行为 | 回归节点（模块一） |
| --- | --- | --- | --- | --- |
| DEF-01 | `eval/metrics_engineering.py` 的时延测量；墙钟固定100秒，高精度计时两次差0.002秒，4次测量 | 0.5毫秒/次，含预热共调用6次 | 使用墙钟，返回0 | `whitebox/test_regression_metrics.py::test_latency_uses_monotonic_high_resolution_clock` |
| DEF-02 | `datasets/rail_dataset.py`；扫描后删除深度文件 | 抛出包含路径的FileNotFoundError | 读取返回None后调用astype，抛出AttributeError | `blackbox/test_regression_depth.py::test_depth_read_failure[missing]` |
| DEF-03 | `eval/metrics_engineering.py`；n_run=0 | 执行模型前抛出说明n_run的ValueError | 除零错误 | `whitebox/test_regression_metrics.py::test_latency_invalid_runs[0]` |

潜在修复方向分别为高分辨率单调时钟、读取结果空值检查、执行次数参数校验；本次结构清理不实施修复。

此外，部分参数非法值未在目录扫描前校验；原始断言要求的ValueError与实际FileNotFoundError不符。完整失败节点见 [执行证据](../evidence/README.md)。实测时延断言可能因计时分辨率产生波动，不能把该波动直接归因于目录调整。
