# 必要测试补充（2026-09-15）

## 本次范围和结果

补充21项：采样与划分8项、空间对齐3项、最小集成1项、AI辅助深度性质9项。
全量215项：204通过、0失败、11项因无CUDA跳过，耗时25.12秒。
模块一199项（188通过、11跳过），模块二16项全部通过；不将Hypothesis生成的示例次数计作独立用例。

## 需求与设计依据

| 需求 | 输入与预期 | 自动化实现 |
| --- | --- | --- |
| 数量参数优先于比例、数量不超可用数据 | 10帧指定4帧且比例0.9，仍选4帧；3帧请求20帧仍为3帧 | test_sampling_behavior.py |
| 时间均匀/随机采样及划分隔离 | 检查明确帧号、同种子重现、无重复；训练与验证不交叉且并集等于采样全集 | test_sampling_behavior.py |
| 模态文件匹配 | 缺少depth不纳入样本；reflectance后缀与.tif正确配对 | test_sampling_behavior.py |
| patch和GT空间一致 | 上下两个不同位置的掩码；中心8列裁剪；输出4/8/16像素；分别验证RGB、depth和GT预期矩阵 | test_spatial_alignment.py |
| 批处理到损失的接口连通 | 3帧按2+1成批，双模态拼接送小型Conv2d，有限非零损失和梯度 | test_pipeline.py |
| 深度归一化数学性质 | 常量minmax/zscore为0、全零有限、log1p已知值、单调性、通道相同 | test_depth_properties.py |
| 缓存不改变输入语义 | 改变patch访问顺序、切换预加载仍得到相同数值；符合全帧有效像素统计 | test_depth_properties.py |

最小集成使用小型模型，不代表TRD结构、完整训练或精度验收。

## DEF-20260915-04：懒加载patch的zscore结果依赖访问顺序

- 严重程度：高；输入数值会随访问顺序或preload配置改变，破坏训练/评估协议一致性。
- 触发：16×24非均匀深度图（1到384），8×8 patch，stride=8，depth_norm=zscore。
- 原行为：懒加载先裁剪，再把第一个patch统计量缓存到整张图路径下；预加载却使用整帧统计。
- 预期：依据既有预加载约定，所有patch使用全帧depth>0像素的mean/std；访问顺序和加载策略不改变数值。
- 复现：test_zscore_patch_independent_of_access_order、test_zscore_patch_matches_full_frame_statistics、test_patch_preload_matches_lazy；修复前三项全部失败，另18项新增测试通过。
- 修复：_load_depth在patch裁剪前计算并缓存全帧统计，全零回退与预加载一致；minmax/log保持原处理范围。
- 验证：上述三项及完整215项测试通过/按设备跳过，原始证据见 logs/supplement-20260915/before.log 与 after.log。
- 兼容性：返回结构、dtype和checkpoint格式不变。旧版懒加载patch输入数值会改变；旧模型或历史分数若依赖该缺陷，必须重新评估，不能直接混用旧结果。

## AI辅助过程与反思

根据前轮覆盖缺口提出数学性质及加载策略等价关系，生成测试后先运行被测旧行为，三个独立断言共同定位同一缓存缺陷，再修复并全量回归。
Hypothesis仅探索有限输入范围，不证明所有输入正确；明确的数值预期和变形关系比随机参数数量更重要。新9项功能测试与已有7项组成模块二16个独立函数，仍需转换为正式Excel清单，并由小组完善AI对话记录和人工审查说明。

## 复现与范围限制

命令：从仓库根目录使用 AI_common 运行 `python -m pytest --cov --cov-config=.coveragerc --cov-report=term-missing`（python必须为项目指定解释器）。
实际日志及JUnit：tests/logs/supplement-20260915/。environment.json记录Python、关键依赖版本、Git基准与被测源码/测试SHA256，标记当前未提交工作区。
报告综合覆盖：数据集76.39%、工程指标89.66%、损失函数100%，三个指定模块合计81.00%；不是整个项目覆盖率。
CUDA、完整模型训练、真实数据精度及模型权重兼容运行未验证；没有下载权重、启动大规模训练。
正式Excel/Word/PPT/演示视频及成员贡献仍待完成。

## 本轮新增执行节点

| pytest节点 | 实际结果 |
| --- | --- |
| tests.module1.blackbox.test_sampling_behavior::test_count_overrides_ratio | 通过 |
| tests.module1.blackbox.test_sampling_behavior::test_requested_count_capped_by_available | 通过 |
| tests.module1.blackbox.test_sampling_behavior::test_ratio_selects_actual_frames | 通过 |
| tests.module1.blackbox.test_sampling_behavior::test_random_sampling_reproducible_unique | 通过 |
| tests.module1.blackbox.test_sampling_behavior::test_sampled_train_val_partition_is_complete[random] | 通过 |
| tests.module1.blackbox.test_sampling_behavior::test_sampled_train_val_partition_is_complete[uniform_time] | 通过 |
| tests.module1.blackbox.test_sampling_behavior::test_missing_pair_is_excluded | 通过 |
| tests.module1.blackbox.test_sampling_behavior::test_reflectance_and_tif_matching | 通过 |
| tests.module1.blackbox.test_spatial_alignment::test_rgb_depth_gt_alignment[4] | 通过 |
| tests.module1.blackbox.test_spatial_alignment::test_rgb_depth_gt_alignment[8] | 通过 |
| tests.module1.blackbox.test_spatial_alignment::test_rgb_depth_gt_alignment[16] | 通过 |
| tests.module1.test_pipeline::test_dataset_batch_loss_backward | 通过 |
| tests.module2.ai_assisted.test_depth_properties::test_minmax_constant_is_zero | 通过 |
| tests.module2.ai_assisted.test_depth_properties::test_zscore_constant_is_finite_zero | 通过 |
| tests.module2.ai_assisted.test_depth_properties::test_all_zero_depth_is_finite | 通过 |
| tests.module2.ai_assisted.test_depth_properties::test_log_matches_known_value | 通过 |
| tests.module2.ai_assisted.test_depth_properties::test_minmax_range_and_order | 通过 |
| tests.module2.ai_assisted.test_depth_properties::test_zscore_patch_independent_of_access_order | 通过 |
| tests.module2.ai_assisted.test_depth_properties::test_zscore_patch_matches_full_frame_statistics | 通过 |
| tests.module2.ai_assisted.test_depth_properties::test_patch_preload_matches_lazy | 通过 |
| tests.module2.ai_assisted.test_depth_properties::test_depth_channels_are_identical | 通过 |
