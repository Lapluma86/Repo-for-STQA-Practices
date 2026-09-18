# 测试设计与用例

模块一采用人工设计的等价类、边界值、判定表和状态转换，以及白盒语句/分支/条件/路径设计；AI辅助回归用例保留在对应模块，设计来源不等同执行方式。所有用例均由pytest运行。

当前模块一48项：黑盒24、白盒23、集成1。下表保留稳定编号，结果来自2026-09-15 CPU证据；本次复核结果见CURRENT_STATUS.md，编号空缺不补号。每个节点对应源码中的输入构造、断言与预期。

| 编号 | 类别 | pytest节点 | 本次结果 |
| --- | --- | --- | --- |
| M1-S01 | 边界值 | `tests/module1/blackbox/test_boundary_value.py::TestBoundaryValueAnalysis::test_img_size_minimum_value` | 通过 |
| M1-S02 | 边界值 | `tests/module1/blackbox/test_boundary_value.py::TestBoundaryValueAnalysis::test_img_size_zero_boundary` | 失败 |
| M1-S03 | 边界值 | `tests/module1/blackbox/test_boundary_value.py::TestBoundaryValueAnalysis::test_patch_stride_zero_invalid` | 失败 |
| M1-S04 | 边界值 | `tests/module1/blackbox/test_boundary_value.py::TestBoundaryValueAnalysis::test_train_sample_ratio_zero` | 通过 |
| M1-S05 | 边界值 | `tests/module1/blackbox/test_boundary_value.py::TestBoundaryValueAnalysis::test_train_sample_ratio_one` | 通过 |
| M1-S06 | 边界值 | `tests/module1/blackbox/test_boundary_value.py::TestBoundaryValueAnalysis::test_view_id_maximum` | 通过 |
| M1-S07 | 判定表 | `tests/module1/blackbox/test_decision_table.py::TestDecisionTable::test_combination_patch_true_split_train` | 通过 |
| M1-S08 | 判定表 | `tests/module1/blackbox/test_decision_table.py::TestDecisionTable::test_combination_patch_false_split_test` | 通过 |
| M1-S09 | 判定表 | `tests/module1/blackbox/test_decision_table.py::TestDecisionTable::test_combination_log_patch_true` | 通过 |
| M1-S10 | 判定表 | `tests/module1/blackbox/test_decision_table.py::TestDecisionTable::test_combination_sample_ratio_with_num` | 通过 |
| M1-S11 | 等价类 | `tests/module1/blackbox/test_equivalence_partitioning.py::TestEquivalencePartitioning::test_view_id_valid_range` | 通过 |
| M1-S12 | 等价类 | `tests/module1/blackbox/test_equivalence_partitioning.py::TestEquivalencePartitioning::test_view_id_invalid_zero` | 失败 |
| M1-S13 | 等价类 | `tests/module1/blackbox/test_equivalence_partitioning.py::TestEquivalencePartitioning::test_split_valid_train` | 通过 |
| M1-S14 | 等价类 | `tests/module1/blackbox/test_equivalence_partitioning.py::TestEquivalencePartitioning::test_split_invalid_unknown` | 失败 |
| M1-S15 | 等价类 | `tests/module1/blackbox/test_equivalence_partitioning.py::TestEquivalencePartitioning::test_depth_norm_valid_minmax` | 通过 |
| M1-S16 | 等价类 | `tests/module1/blackbox/test_equivalence_partitioning.py::TestEquivalencePartitioning::test_depth_norm_invalid` | 失败 |
| M1-S17 | 等价类 | `tests/module1/blackbox/test_equivalence_partitioning.py::TestEquivalencePartitioning::test_train_sample_ratio_invalid_greater_than_one` | 失败 |
| M1-S18 | 等价类 | `tests/module1/blackbox/test_regression_depth.py::test_depth_read_failure[missing]` | 失败 |
| M1-S19 | 场景法 | `tests/module1/blackbox/test_sampling_behavior.py::test_count_overrides_ratio` | 通过 |
| M1-S20 | 场景法 | `tests/module1/blackbox/test_sampling_behavior.py::test_random_sampling_reproducible_unique` | 通过 |
| M1-S21 | 场景法 | `tests/module1/blackbox/test_spatial_alignment.py::test_rgb_depth_gt_alignment[8]` | 通过 |
| M1-S22 | 场景法 | `tests/module1/blackbox/test_state_transition.py::test_scan_then_read` | 通过 |
| M1-S23 | 场景法 | `tests/module1/blackbox/test_state_transition.py::test_lazy_preload_equivalence` | 通过 |
| M1-S24 | 场景法 | `tests/module1/blackbox/test_state_transition.py::test_state_transition_error_recovery` | 通过 |
| M1-S25 | 场景法 | `tests/module1/test_pipeline.py::test_dataset_batch_loss_backward` | 通过 |
| M1-S26 | 分支覆盖 | `tests/module1/whitebox/test_branch_coverage.py::TestBranchCoverage::test_count_trainable_params_total_greater_than_zero_true` | 通过 |
| M1-S27 | 分支覆盖 | `tests/module1/whitebox/test_branch_coverage.py::TestBranchCoverage::test_count_trainable_params_total_greater_than_zero_false` | 通过 |
| M1-S29 | 分支覆盖 | `tests/module1/whitebox/test_branch_coverage.py::TestBranchCoverage::test_compute_fp_per_image_n_normal_zero_true` | 通过 |
| M1-S30 | 分支覆盖 | `tests/module1/whitebox/test_branch_coverage.py::TestBranchCoverage::test_compute_fp_per_image_n_normal_zero_false` | 通过 |
| M1-S31 | 分支覆盖 | `tests/module1/whitebox/test_branch_coverage.py::TestBranchCoverage::test_loss_distil_pixel_all_branches` | 通过 |
| M1-S32 | 条件覆盖 | `tests/module1/whitebox/test_condition_coverage.py::TestConditionCoverage::test_condition_i_equals_0_true_i_equals_1_false` | 通过 |
| M1-S33 | 条件覆盖 | `tests/module1/whitebox/test_condition_coverage.py::TestConditionCoverage::test_condition_i_equals_0_false_i_equals_1_true` | 通过 |
| M1-S34 | 条件覆盖 | `tests/module1/whitebox/test_condition_coverage.py::TestConditionCoverage::test_condition_i_equals_0_false_i_equals_1_false` | 通过 |
| M1-S35 | 条件覆盖 | `tests/module1/whitebox/test_condition_coverage.py::TestConditionCoverage::test_condition_isinstance_tuple_true` | 失败 |
| M1-S36 | 条件覆盖 | `tests/module1/whitebox/test_condition_coverage.py::TestConditionCoverage::test_condition_isinstance_both_false` | 通过 |
| M1-S37 | 数值校验 | `tests/module1/whitebox/test_loss_values.py::test_l2_multiscale_value_and_gradient` | 通过 |
| M1-S39 | 路径覆盖 | `tests/module1/whitebox/test_path_coverage.py::TestPathCoverage::test_measure_inference_latency_path_1` | 通过 |
| M1-S40 | 路径覆盖 | `tests/module1/whitebox/test_path_coverage.py::TestPathCoverage::test_loss_distil_pixel_path_5` | 通过 |
| M1-S41 | 路径覆盖 | `tests/module1/whitebox/test_path_coverage.py::TestPathCoverage::test_loss_l2_path_zero_iterations` | 通过 |
| M1-S42 | 路径覆盖 | `tests/module1/whitebox/test_path_coverage.py::TestPathCoverage::test_loss_l2_path_multiple_iterations` | 通过 |
| M1-S43 | 边界值 | `tests/module1/whitebox/test_regression_metrics.py::test_latency_uses_monotonic_high_resolution_clock` | 失败 |
| M1-S44 | 边界值 | `tests/module1/whitebox/test_regression_metrics.py::test_latency_invalid_runs[0]` | 失败 |
| M1-S45 | 语句覆盖 | `tests/module1/whitebox/test_statement_coverage.py::TestStatementCoverage::test_loss_l2_basic` | 通过 |
| M1-S46 | 语句覆盖 | `tests/module1/whitebox/test_statement_coverage.py::TestStatementCoverage::test_loss_distil_basic` | 通过 |
| M1-S47 | 语句覆盖 | `tests/module1/whitebox/test_statement_coverage.py::TestStatementCoverage::test_loss_distil_p_basic` | 通过 |
| M1-S48 | 语句覆盖 | `tests/module1/whitebox/test_statement_coverage.py::TestStatementCoverage::test_calculate_pixel_similarity_f_2_basic` | 通过 |
| M1-S49 | 语句覆盖 | `tests/module1/whitebox/test_statement_coverage.py::TestStatementCoverage::test_count_trainable_params_frozen_layers` | 通过 |
| M1-S50 | 语句覆盖 | `tests/module1/whitebox/test_statement_coverage.py::TestStatementCoverage::test_compute_fp_per_image_list_input` | 通过 |

## 模块二：AI辅助测试

共16项，利用Hypothesis探索参数组合及数值/变形性质，人工核对断言与接口约定。参数组合的可执行断言以测试源码为准。

| 测试文件 | 函数 | 设计说明 |
| --- | --- | --- |
| test_depth_properties.py | `test_minmax_constant_is_zero` | 验证函数名所述参数或数学性质；输入与预期见测试断言。 |
| test_depth_properties.py | `test_zscore_constant_is_finite_zero` | 验证函数名所述参数或数学性质；输入与预期见测试断言。 |
| test_depth_properties.py | `test_all_zero_depth_is_finite` | 验证函数名所述参数或数学性质；输入与预期见测试断言。 |
| test_depth_properties.py | `test_log_matches_known_value` | 验证函数名所述参数或数学性质；输入与预期见测试断言。 |
| test_depth_properties.py | `test_minmax_range_and_order` | 验证函数名所述参数或数学性质；输入与预期见测试断言。 |
| test_depth_properties.py | `test_zscore_patch_independent_of_access_order` | 验证函数名所述参数或数学性质；输入与预期见测试断言。 |
| test_depth_properties.py | `test_zscore_patch_matches_full_frame_statistics` | 验证函数名所述参数或数学性质；输入与预期见测试断言。 |
| test_depth_properties.py | `test_patch_preload_matches_lazy` | 验证函数名所述参数或数学性质；输入与预期见测试断言。 |
| test_depth_properties.py | `test_depth_channels_are_identical` | 验证函数名所述参数或数学性质；输入与预期见测试断言。 |
| test_parameter_validation.py | `test_parameter_combinations` | Hypothesis生成view_id、split与img_size组合，核对构造和错误处理 |
| test_parameter_validation.py | `test_sampling_edge_cases` | 探索采样比例、视角与patch步长边界 |
| test_parameter_validation.py | `test_config_combinations` | 探索归一化、patch、预加载和采样模式组合 |
| test_parameter_validation.py | `test_patch_stride_zero_still_rejected` | 验证零patch步长应被拒绝；当前冻结源码尚未修复 |
| test_parameter_validation.py | `test_invalid_view_id_still_rejected` | 验证非法视角应被拒绝；检查异常类型与触发顺序 |
| test_parameter_validation.py | `test_invalid_depth_norm_still_rejected` | 验证非法归一化模式应被拒绝 |
| test_parameter_validation.py | `test_empty_list_in_loss_functions` | 验证空特征列表的损失函数行为 |
