# Scripts 目录说明

按职责整理后的脚本布局如下：

- `data/`
  - 数据审计、离线预处理、测试集构建
- `diagnostics/`
  - 轻量级数据/流程自检脚本、分支诊断脚本
- `eval/`
  - 单模型评估、批量评估
- `maintenance/`
  - 输出目录整理、实验归档
- `train/`
  - 当前仍在使用的训练与 PEFT 脚本
- `archive/`
  - 已不建议直接使用的旧启动脚本、占位脚本

## 当前推荐入口

- 批量训练：`scripts/train/train_all_views.sh`
- 单 ckpt 评估：`scripts/eval/eval_from_ckpt.py`
- rail_all 批量补评估：`scripts/eval/eval_rail_all.py`
- Cam4 P1 训练：`scripts/train/train_cam4_depth_p1_cv.py`
- 扩充 Cam4 测试集：`scripts/data/build_augmented_cam4_testset.py`
- 分支 AUROC / 真隔离诊断：`scripts/diagnostics/eval_branch_auroc.py`
- 逐图排序变化分析：`scripts/diagnostics/analyze_branch_rank_changes.py`
- 关键帧导出与 depth 预览：`scripts/diagnostics/export_key_frames.py`
- 关键帧 anomaly map 并排可视化：`scripts/diagnostics/export_key_frame_anomaly_maps.py`
- P1 论文 qualitative 拼图：`scripts/diagnostics/make_p1_qualitative_figure.py`
- 整理 `outputs/rail_all`：`scripts/maintenance/organize_rail_all.py`
- 整理 `outputs/rail_peft`：`scripts/maintenance/organize_rail_peft_runs.py`
