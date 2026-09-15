> 当前结论已更新：请参阅 [CURRENT_STATUS.md](CURRENT_STATUS.md) 和 [DEFECT_AUDIT.md](DEFECT_AUDIT.md)。以下保留历史原文。

> 历史文档：本文保留迁移前的阶段名称、命令和统计，可能已过期。当前目录、运行方式及课程要求以仓库根目录 `tests/README.md` 为准；原 `manual/phase1` 已迁至 `module1`，原 `manual/phase2` 已迁至 `module2`。

# SteelRailWay 测试 - 完整报告

## 项目信息

| 项目 | 内容 |
|------|-------|
| 项目名称 | 铁路缺陷检测系统测试 |
| 完成日期 | 2026-09-11 |
| 测试用例总数 | 179 |
| 测试结果 | 168个通过，11个跳过 |

---

## 测试总结

### 模块一 - 第一阶段：手工测试

| 类别 | 用例数 | 结果 |
|----------|-------|--------|
| 黑盒测试 - 等价类划分 | 27 | 通过 |
| 黑盒测试 - 边界值分析 | 27 | 通过 |
| 黑盒测试 - 判定表测试 | 16 | 通过 |
| 黑盒测试 - 状态转换测试 | 18 | 通过 |
| 白盒测试 - 语句覆盖 | 24 | 通过 |
| 白盒测试 - 分支覆盖 | 20 | 通过 |
| 白盒测试 - 条件覆盖 | 18 | 通过 |
| 白盒测试 - 路径覆盖 | 22 | 通过 |
| **第一阶段总计** | **172** | **161个通过，11个跳过** |

### 模块一 - 第二阶段：AI辅助测试

| 类别 | 用例数 | 结果 |
|----------|-------|--------|
| 参数验证（Hypothesis） | 1 | 通过 |
| 边界情况发现（Hypothesis） | 1 | 通过 |
| 配置组合（Hypothesis） | 1 | 通过 |
| 回归测试 | 4 | 通过 |
| **第二阶段总计** | **7** | **7个通过** |

---

## 测试执行

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定模块
pytest tests/manual/phase1/ -v
pytest tests/manual/phase2/ -v

# 运行并生成覆盖率报告
pytest tests/manual/phase1/whitebox/ \
    --cov=utils.losses \
    --cov=eval.metrics_engineering \
    --cov-report=html:tests/manual/phase1/coverage_reports
```

### 结果

```
==================== 168个通过，11个跳过，耗时5.34秒 ====================
```

---

## 覆盖率报告

| 模块 | 覆盖率 |
|--------|----------|
| `utils/losses.py` | 97% |
| `eval/metrics_engineering.py` | 92% |
| **总体** | **96%** |

---

## 已修复的缺陷

| ID | 优先级 | 描述 | 文件 |
|----|---------|-------------|------|
| DEF-004 | 高 | patch_stride=0 未验证 | `datasets/rail_dataset.py` |
| DEF-001 | 中 | 输入参数缺乏验证 | `datasets/rail_dataset.py` |
| DEF-003 | 中 | depth_norm 参数未验证 | `datasets/rail_dataset.py` |
| DEF-WB-003 | 中 | 除零保护 | `eval/metrics_engineering.py` |
| DEF-002 | 低 | 路径参数未验证 | `datasets/rail_dataset.py` |
| DEF-WB-001 | 低 | 空列表处理 | `utils/losses.py` |

---

## 已添加的参数验证

| 参数 | 验证规则 |
|-----------|------------|
| `view_id` | 1-8 范围 |
| `split` | "train" \| "val" \| "test" |
| `img_size` | 正整数 |
| `depth_norm` | "zscore" \| "minmax" \| "log" |
| `patch_size` | 正整数 |
| `patch_stride` | 正整数（>0） |
| `train_sample_ratio` | 0-1 范围 |
| `train_sample_num` | 正整数 |
| `preload_workers` | 正整数 |
| `sampling_mode` | "random" \| "uniform_time" |

---

## 下一步工作

1. 模块二：自动化测试
2. 模块二：性能测试
3. 模块二：安全测试
4. 模块二：集成测试
