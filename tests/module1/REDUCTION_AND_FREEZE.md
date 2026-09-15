# 模块一：50项用例与源码冻结

## 范围及精简原则

按用户要求，模块一从199项精简至50项（保留25.13%）；模块二16项不参与本次缩减，默认全量收集共66项。
直接删除多余测试函数并缩减三个参数化列表，没有通过 deselect、skip 或 xfail 隐藏用例。原测试完整内容可由提交 `8da48445d393e882135e659a2b8b72eddee5b0cd` 恢复，逐文件保留/删除记录见 [selection.json](selection.json)。

按原文件类别约四分之一选取，并照顾小类别至少一项及三项指定缺陷。优先保留有效/无效输入、关键边界、条件真值、正常/异常路径、真实数据读取和数值验证；不以运行是否通过决定取舍。
黑盒95→24（47.74%→48%），白盒103→25（51.76%→50%），集成1→1（0.50%→2%）。
覆盖设计类别仍然保留，但不声称精简后已达到完整语句、分支、条件或路径覆盖。

| 类别 | 原数量 | 当前数量 | 保留比例 |
| --- | ---: | ---: | ---: |
| 边界值 | 27 | 6 | 22.2% |
| 判定表 | 16 | 4 | 25.0% |
| 等价类 | 27 | 7 | 25.9% |
| 深度读取缺陷 | 2 | 1 | 50.0% |
| 采样行为 | 8 | 2 | 25.0% |
| 空间对齐 | 3 | 1 | 33.3% |
| 状态/生命周期 | 12 | 3 | 25.0% |
| 最小集成 | 1 | 1 | 100.0% |
| 分支覆盖设计 | 24 | 6 | 25.0% |
| 条件覆盖设计 | 20 | 5 | 25.0% |
| 损失数值/梯度 | 5 | 1 | 20.0% |
| 路径覆盖设计 | 22 | 5 | 22.7% |
| 时延缺陷 | 8 | 2 | 25.0% |
| 语句覆盖设计 | 24 | 6 | 25.0% |

## 源码回退和冻结

三个文件逐字节恢复至首次课程测试提交 `976d37d` 的父提交 `e05aafbf8dcf320622b04109d3a8766229cea94f`：

- `SteelRailWay/datasets/rail_dataset.py`：撤销参数/路径/patch校验、深度读取修复、zscore缓存修复。
- `SteelRailWay/eval/metrics_engineering.py`：撤销高分辨率计时和测量次数校验，并恢复原导入。
- `SteelRailWay/utils/losses.py`：撤销空特征列表处理。

已检查产品文件历史，测试期间的产品修改集中于这三个文件。其他产品源码保持现有状态，不覆盖已有实验产物或此前的测试目录迁移。
[source-freeze.json](../source-freeze.json) 记录当前86个受版本管理的产品Python文件指纹；换行统一为LF后计算SHA-256。
`tests/conftest.py` 在每次pytest收集前校验文件，缺失或内容变化会终止执行，不占用50项用例配额。
已验证原文件允许执行、模拟内容变更被拒绝。冻结是校验约束，不设置操作系统只读属性；新增文件不在现有指纹清单内。
后续课程测试应保持此基线，不能为使测试通过而修改产品源码或重生成指纹。

## 本次模块一执行结果

2026-09-15，AI_common / Python 3.10.20 / CPU：**50项，39通过、10失败、1跳过、0执行错误**，退出码1，耗时32.10秒。
CUDA峰值显存用例因无设备跳过。首次沙箱运行发生临时目录权限错误，未作为产品缺陷证据；本表来自获准在沙箱外的完整执行。

三指定模块的语句/分支综合覆盖率为70.78%：数据集65.29%，工程指标88.00%，损失函数89.77%。只代表本次模块一和明确的三个模块，不是全项目覆盖率，也不能与源码修复后的历史覆盖率直接比较。

三个指定缺陷各保留一项，均真实失败：

| 缺陷 | 测试 | 预期 | 本次实际 |
| --- | --- | --- | --- |
| DEF-20260915-01 | test_latency_uses_monotonic_high_resolution_clock | 0.5毫秒 | 0毫秒，断言失败 |
| DEF-20260915-02 | test_depth_read_failure[missing] | 带路径的FileNotFoundError | AttributeError（None.astype） |
| DEF-20260915-03 | test_latency_invalid_runs[0] | n_run参数ValueError | ZeroDivisionError |

另6项无效参数用例（img_size=0、patch_stride=0、view_id=0、未知split、未知depth_norm、采样比例>1）因旧版未拒绝参数而失败。
`test_condition_isinstance_tuple_true` 实测时延为0，与DEF-20260915-01属于同一问题，不另计独立缺陷；该真实时钟用例在不同机器上可能通过，确定性的指定缺陷用例仍提供稳定复现。
共10个失败节点不等于10个独立缺陷。所有原预期断言保留，无xfail包装，也未将异常断言改为接受当前错误行为。
历史修复前后证据继续保存；当前三项缺陷状态为“未修复/已复现”，不得表述为本版本已完成修复验证。

## 模块二影响

模块二保持16项未精简，单独执行得到6通过、10失败、0跳过，耗时60.01秒，退出码1。失败涉及归一化缓存、参数校验及空列表返回类型；不自动等同10个独立缺陷。见[模块二日志](../logs/frozen-50/module2.log)及[当前状态](../CURRENT_STATUS.md)。

## 复现

从仓库根目录运行：

```powershell
& 'D:\Anaconda\envs\AI_common\python.exe' -m pytest -c pytest.ini tests/module1/ --junitxml=tests/artifacts/module1-frozen.xml

# 单独演示三个指定缺陷（收集3项）
& 'D:\Anaconda\envs\AI_common\python.exe' -m pytest -c pytest.ini tests/module1/blackbox/test_regression_depth.py tests/module1/whitebox/test_regression_metrics.py
```

完整执行证据：[日志](../logs/frozen-50/module1.log)、[JUnit](../logs/frozen-50/module1.xml)、[HTML](../logs/frozen-50/module1.html)、[覆盖率JSON](../logs/frozen-50/coverage.json)、[新旧用例映射](../logs/frozen-50/case-index.json)。
旧Excel、Word、PPT及199项报告保留为历史资料，本次未重新生成办公附件；它们不能用于当前50项冻结版本的提交验收。
未执行训练、真实数据精度评估、CUDA实测。

## 当前50项清单

原编号来自历史交付清单，用于追溯设计；当前编号不改变原函数的输入及预期。

| 当前编号 | 原编号 | 类别 | pytest节点 | 本次结果 |
| --- | --- | --- | --- | --- |
| M1-S01 | M1-001 | 边界值 | `tests/module1/blackbox/test_boundary_value.py::TestBoundaryValueAnalysis::test_img_size_minimum_value` | 通过 |
| M1-S02 | M1-005 | 边界值 | `tests/module1/blackbox/test_boundary_value.py::TestBoundaryValueAnalysis::test_img_size_zero_boundary` | 失败 |
| M1-S03 | M1-011 | 边界值 | `tests/module1/blackbox/test_boundary_value.py::TestBoundaryValueAnalysis::test_patch_stride_zero_invalid` | 失败 |
| M1-S04 | M1-014 | 边界值 | `tests/module1/blackbox/test_boundary_value.py::TestBoundaryValueAnalysis::test_train_sample_ratio_zero` | 通过 |
| M1-S05 | M1-018 | 边界值 | `tests/module1/blackbox/test_boundary_value.py::TestBoundaryValueAnalysis::test_train_sample_ratio_one` | 通过 |
| M1-S06 | M1-023 | 边界值 | `tests/module1/blackbox/test_boundary_value.py::TestBoundaryValueAnalysis::test_view_id_maximum` | 通过 |
| M1-S07 | M1-028 | 判定表 | `tests/module1/blackbox/test_decision_table.py::TestDecisionTable::test_combination_patch_true_split_train` | 通过 |
| M1-S08 | M1-031 | 判定表 | `tests/module1/blackbox/test_decision_table.py::TestDecisionTable::test_combination_patch_false_split_test` | 通过 |
| M1-S09 | M1-034 | 判定表 | `tests/module1/blackbox/test_decision_table.py::TestDecisionTable::test_combination_log_patch_true` | 通过 |
| M1-S10 | M1-040 | 判定表 | `tests/module1/blackbox/test_decision_table.py::TestDecisionTable::test_combination_sample_ratio_with_num` | 通过 |
| M1-S11 | M1-044 | 等价类 | `tests/module1/blackbox/test_equivalence_partitioning.py::TestEquivalencePartitioning::test_view_id_valid_range` | 通过 |
| M1-S12 | M1-047 | 等价类 | `tests/module1/blackbox/test_equivalence_partitioning.py::TestEquivalencePartitioning::test_view_id_invalid_zero` | 失败 |
| M1-S13 | M1-050 | 等价类 | `tests/module1/blackbox/test_equivalence_partitioning.py::TestEquivalencePartitioning::test_split_valid_train` | 通过 |
| M1-S14 | M1-054 | 等价类 | `tests/module1/blackbox/test_equivalence_partitioning.py::TestEquivalencePartitioning::test_split_invalid_unknown` | 失败 |
| M1-S15 | M1-060 | 等价类 | `tests/module1/blackbox/test_equivalence_partitioning.py::TestEquivalencePartitioning::test_depth_norm_valid_minmax` | 通过 |
| M1-S16 | M1-062 | 等价类 | `tests/module1/blackbox/test_equivalence_partitioning.py::TestEquivalencePartitioning::test_depth_norm_invalid` | 失败 |
| M1-S17 | M1-067 | 等价类 | `tests/module1/blackbox/test_equivalence_partitioning.py::TestEquivalencePartitioning::test_train_sample_ratio_invalid_greater_than_one` | 失败 |
| M1-S18 | M1-071 | 等价类 | `tests/module1/blackbox/test_regression_depth.py::test_depth_read_failure[missing]` | 失败 |
| M1-S19 | M1-073 | 场景法 | `tests/module1/blackbox/test_sampling_behavior.py::test_count_overrides_ratio` | 通过 |
| M1-S20 | M1-076 | 场景法 | `tests/module1/blackbox/test_sampling_behavior.py::test_random_sampling_reproducible_unique` | 通过 |
| M1-S21 | M1-082 | 场景法 | `tests/module1/blackbox/test_spatial_alignment.py::test_rgb_depth_gt_alignment[8]` | 通过 |
| M1-S22 | M1-086 | 场景法 | `tests/module1/blackbox/test_state_transition.py::test_scan_then_read` | 通过 |
| M1-S23 | M1-091 | 场景法 | `tests/module1/blackbox/test_state_transition.py::test_lazy_preload_equivalence` | 通过 |
| M1-S24 | M1-095 | 场景法 | `tests/module1/blackbox/test_state_transition.py::test_state_transition_error_recovery` | 通过 |
| M1-S25 | M1-096 | 场景法 | `tests/module1/test_pipeline.py::test_dataset_batch_loss_backward` | 通过 |
| M1-S26 | M1-097 | 分支覆盖 | `tests/module1/whitebox/test_branch_coverage.py::TestBranchCoverage::test_count_trainable_params_total_greater_than_zero_true` | 通过 |
| M1-S27 | M1-098 | 分支覆盖 | `tests/module1/whitebox/test_branch_coverage.py::TestBranchCoverage::test_count_trainable_params_total_greater_than_zero_false` | 通过 |
| M1-S28 | M1-102 | 分支覆盖 | `tests/module1/whitebox/test_branch_coverage.py::TestBranchCoverage::test_measure_peak_gpu_memory_not_cuda_false` | 跳过 |
| M1-S29 | M1-103 | 分支覆盖 | `tests/module1/whitebox/test_branch_coverage.py::TestBranchCoverage::test_compute_fp_per_image_n_normal_zero_true` | 通过 |
| M1-S30 | M1-104 | 分支覆盖 | `tests/module1/whitebox/test_branch_coverage.py::TestBranchCoverage::test_compute_fp_per_image_n_normal_zero_false` | 通过 |
| M1-S31 | M1-108 | 分支覆盖 | `tests/module1/whitebox/test_branch_coverage.py::TestBranchCoverage::test_loss_distil_pixel_all_branches` | 通过 |
| M1-S32 | M1-121 | 条件覆盖 | `tests/module1/whitebox/test_condition_coverage.py::TestConditionCoverage::test_condition_i_equals_0_true_i_equals_1_false` | 通过 |
| M1-S33 | M1-122 | 条件覆盖 | `tests/module1/whitebox/test_condition_coverage.py::TestConditionCoverage::test_condition_i_equals_0_false_i_equals_1_true` | 通过 |
| M1-S34 | M1-123 | 条件覆盖 | `tests/module1/whitebox/test_condition_coverage.py::TestConditionCoverage::test_condition_i_equals_0_false_i_equals_1_false` | 通过 |
| M1-S35 | M1-127 | 条件覆盖 | `tests/module1/whitebox/test_condition_coverage.py::TestConditionCoverage::test_condition_isinstance_tuple_true` | 失败 |
| M1-S36 | M1-129 | 条件覆盖 | `tests/module1/whitebox/test_condition_coverage.py::TestConditionCoverage::test_condition_isinstance_both_false` | 通过 |
| M1-S37 | M1-141 | 数值校验 | `tests/module1/whitebox/test_loss_values.py::test_l2_multiscale_value_and_gradient` | 通过 |
| M1-S38 | M1-148 | 路径覆盖 | `tests/module1/whitebox/test_path_coverage.py::TestPathCoverage::test_measure_peak_gpu_memory_path_1` | 通过 |
| M1-S39 | M1-152 | 路径覆盖 | `tests/module1/whitebox/test_path_coverage.py::TestPathCoverage::test_measure_inference_latency_path_1` | 通过 |
| M1-S40 | M1-159 | 路径覆盖 | `tests/module1/whitebox/test_path_coverage.py::TestPathCoverage::test_loss_distil_pixel_path_5` | 通过 |
| M1-S41 | M1-160 | 路径覆盖 | `tests/module1/whitebox/test_path_coverage.py::TestPathCoverage::test_loss_l2_path_zero_iterations` | 通过 |
| M1-S42 | M1-162 | 路径覆盖 | `tests/module1/whitebox/test_path_coverage.py::TestPathCoverage::test_loss_l2_path_multiple_iterations` | 通过 |
| M1-S43 | M1-168 | 边界值 | `tests/module1/whitebox/test_regression_metrics.py::test_latency_uses_monotonic_high_resolution_clock` | 失败 |
| M1-S44 | M1-169 | 边界值 | `tests/module1/whitebox/test_regression_metrics.py::test_latency_invalid_runs[0]` | 失败 |
| M1-S45 | M1-176 | 语句覆盖 | `tests/module1/whitebox/test_statement_coverage.py::TestStatementCoverage::test_loss_l2_basic` | 通过 |
| M1-S46 | M1-179 | 语句覆盖 | `tests/module1/whitebox/test_statement_coverage.py::TestStatementCoverage::test_loss_distil_basic` | 通过 |
| M1-S47 | M1-181 | 语句覆盖 | `tests/module1/whitebox/test_statement_coverage.py::TestStatementCoverage::test_loss_distil_p_basic` | 通过 |
| M1-S48 | M1-187 | 语句覆盖 | `tests/module1/whitebox/test_statement_coverage.py::TestStatementCoverage::test_calculate_pixel_similarity_f_2_basic` | 通过 |
| M1-S49 | M1-189 | 语句覆盖 | `tests/module1/whitebox/test_statement_coverage.py::TestStatementCoverage::test_count_trainable_params_frozen_layers` | 通过 |
| M1-S50 | M1-199 | 语句覆盖 | `tests/module1/whitebox/test_statement_coverage.py::TestStatementCoverage::test_compute_fp_per_image_list_input` | 通过 |
