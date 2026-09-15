# 当前测试质量状态（2026-09-15）

本文件为当前结论入口；2026-09-14及更早报告保留为历史证据，不再代表当前结果。

## 执行结果

| 模块 | 收集 | 通过 | 失败 | 跳过 |
| --- | ---: | ---: | ---: | ---: |
| 模块一 | 199 | 188 | 0 | 11 |
| 模块二 | 16 | 16 | 0 | 0 |
| 合计 | 215 | 204 | 0 | 11 |

Windows / AI_common / Python 3.10.20 / PyTorch 2.12.0+cpu，耗时25.12秒，退出码0。
11项跳过均为CUDA相关测试，不视为通过。Hypothesis示例次数不计作独立课程用例条数。

```powershell
Set-Location 'D:\projects\Repo-for-STQA-Practices'
& 'D:\Anaconda\envs\AI_common\python.exe' -m pytest --cov --cov-config=.coveragerc --cov-report=term-missing --html=tests/artifacts/supplement-20260915/after.html --self-contained-html --junitxml=tests/artifacts/supplement-20260915/after.xml
```

纳入版本管理的执行证据位于 [logs/supplement-20260915](logs/supplement-20260915/)。
完整HTML报告在本机 `tests/artifacts/supplement-20260915/after.html`。
本机沙箱的Windows临时目录权限有已知限制，本次使用同一解释器在沙箱外验证。

## 已处理问题

1. 将12项数据生命周期/配置测试改为确定非空的小型真实双模态数据；移除条件跳过断言和宽泛异常吞噬。
2. 按真实接口检查 intensity/depth/gt 的形状、类型、有限值、标签及元数据；校验重复读取、异常后恢复。
3. 配置对比明确命名为配置一致性验证，增加实际训练/验证隔离、确定性划分、预加载与懒加载等价、patch数量及正常/异常GT检查。
4. 时延使用高分辨率单调时钟；拒绝非法测量次数。保留真实时延大于0的原断言，另以可控时钟验证毫秒换算与调用次数。
5. 深度读取先检查失败，再转换类型，保证损坏/丢失文件报告带路径的FileNotFoundError。
6. Windows超长路径用例使用扩展路径，保留实际>200字符路径构造目标。
7. 磁盘/进程启动的Hypothesis功能测试设置deadline=None，仍限制示例数量；不将默认200ms误当课程性能要求。
8. 补充损失精确数值和梯度验证，纠正恒真类型断言。

## 覆盖率

coverage 开启 branch，以下是语句与分支综合统计（包含两个模块的全部测试）：

| 被测模块 | 覆盖率 |
| --- | ---: |
| datasets.rail_dataset | 76.39% |
| eval.metrics_engineering | 89.66% |
| utils.losses | 100.00% |
| 三个指定模块合计 | 81.00% |

相同三模块范围的上一轮综合值为53.88%。不是整个钢轨项目覆盖率，不代表条件/路径覆盖已完全满足。
子进程执行的预加载工作函数未启用子进程覆盖率收集，因此报告中的该函数未覆盖不等于从未执行。

## 尚待完成

- CUDA设备实测；完整训练、真实数据精度与端到端验收本次未执行。
- 数据集剩余路径（多种采样、预resize、异常图像和部分归一化路径等）仍有覆盖缺口。旧参数用例仍有仅检查属性的内容。
- 模块一199项现有pytest节点已整理为正式Excel；Word缺陷清单、测试报告和PPT已完成，见[交付包](../deliverables/module1/README.md)。视频与现场演练形式仍待讨论。
- 成员暂用A/B/C；代码比例已按当前HEAD的Python现存行归属统计，课程总工作量及个人缺陷归属仍需小组据实复核。
- 模块二已有16个独立测试函数，正式Excel用例清单和AI过程交付仍未完成；不能以随机示例数量替代正式用例。

缺陷证据和历史记录审核见 [DEFECT_AUDIT.md](DEFECT_AUDIT.md)。本次代码及文档尚未提交Git。

## 必要测试补充

模块一另行独立执行：188通过、11跳过、0失败，15.55秒；三指定模块语句/分支综合覆盖79.43%。此值不含模块二，证据位于[模块一执行目录](../deliverables/module1/evidence/)。

新增21项、发现并修复patch归一化缓存缺陷。详情与需求映射见 [SUPPLEMENT_REPORT.md](SUPPLEMENT_REPORT.md)。旧版懒加载patch模型结果可能受影响，需重新评估。
