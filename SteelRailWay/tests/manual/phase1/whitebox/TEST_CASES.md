# 白盒测试用例清单 - 模块一第一阶段

## 测试概览
- **项目名称**: SteelRailWay 铁路缺陷检测系统
- **测试阶段**: 模块一 - 第一阶段（人工手写）
- **测试类型**: 白盒测试
- **测试目标模块**: utils/losses.py, eval/metrics_engineering.py
- **测试日期**: 2024年
- **测试人员**: 手工编写

## 测试统计

### 语句覆盖测试 (Statement Coverage)
- **测试文件**: test_statement_coverage.py
- **测试用例数**: 25个
- **目标覆盖率**: 95%+

| 测试用例ID | 测试函数 | 覆盖语句数 | 预期结果 | 状态 |
|-----------|---------|-----------|---------|------|
| TC-SC-001-1 | loss_l2 | 6/6 | 100% | 待测试 |
| TC-SC-001-2 | loss_l2 (空列表) | 6/6 | 100% | 待测试 |
| TC-SC-002-1 | loss_distil | 7/7 | 100% | 待测试 |
| TC-SC-003-1 | loss_distil_p | 7/7 | 100% | 待测试 |
| TC-SC-004-1 | loss_distil_pixel | 14/14 | 100% | 待测试 |
| TC-SC-005-1 | calculate_pixel_similarity | 20/20 | 100% | 待测试 |
| TC-SC-008-1 | count_trainable_params | 5/5 | 100% | 待测试 |
| TC-SC-009-1 | measure_inference_latency | 12/12 | 100% | 待测试 |
| TC-SC-010-1 | measure_peak_gpu_memory | 3/3 | 100% | 待测试 |
| TC-SC-011-1 | compute_fp_per_image | 7/7 | 100% | 待测试 |

**总语句覆盖率目标**: 95%+

### 分支覆盖测试 (Branch Coverage)
- **测试文件**: test_branch_coverage.py
- **测试用例数**: 20个
- **目标覆盖率**: 100%

| 测试用例ID | 分支点 | True分支 | False分支 | 状态 |
|-----------|--------|---------|----------|------|
| TC-BC-001 | total > 0 | ✓ | ✓ | 待测试 |
| TC-BC-002 | use_cuda | ✓ | ✓ | 待测试 |
| TC-BC-003 | device.type != "cuda" | ✓ | ✓ | 待测试 |
| TC-BC-004 | n_normal == 0 | ✓ | ✓ | 待测试 |
| TC-BC-005 | i == 0 or i == 1 | ✓ | ✓ | 待测试 |
| TC-BC-006 | 循环次数 | 0/1/n次 | - | 待测试 |

**总分支覆盖率目标**: 100%

### 条件覆盖测试 (Condition Coverage)
- **测试文件**: test_condition_coverage.py
- **测试用例数**: 18个
- **目标**: 所有子条件都测试True和False

| 测试用例ID | 复合条件 | 子条件1 | 子条件2 | 状态 |
|-----------|---------|---------|---------|------|
| TC-CC-001 | i==0 or i==1 | T/F | T/F | 待测试 |
| TC-CC-002 | total > 0 (三元) | T/F | - | 待测试 |
| TC-CC-003 | isinstance(tuple/list) | T/F | T/F | 待测试 |
| TC-CC-004 | device.type == "cuda" | T/F | - | 待测试 |
| TC-CC-005 | n_normal == 0 | T/F | - | 待测试 |
| TC-CC-006 | 真值表完整性 | 多组合 | - | 待测试 |

**条件覆盖率目标**: 100%

### 路径覆盖测试 (Path Coverage)
- **测试文件**: test_path_coverage.py
- **测试用例数**: 22个
- **覆盖策略**: 基本路径覆盖（基于圈复杂度）

| 测试用例ID | 函数名 | 独立路径数 | 已覆盖 | 状态 |
|-----------|--------|-----------|--------|------|
| TC-PC-001 | count_trainable_params | 2 | 2/2 | 待测试 |
| TC-PC-002 | measure_peak_gpu_memory | 2 | 2/2 | 待测试 |
| TC-PC-003 | compute_fp_per_image | 2 | 2/2 | 待测试 |
| TC-PC-004 | measure_inference_latency | 3 | 3/3 | 待测试 |
| TC-PC-005 | loss_distil_pixel | 5 | 5/5 | 待测试 |
| TC-PC-006 | loss_l2 (循环) | 3 | 3/3 | 待测试 |

**路径覆盖率目标**: 90%+ (主要路径)

## 代码覆盖率目标

### utils/losses.py
- **语句覆盖**: 95%+
- **分支覆盖**: 100%
- **函数覆盖**: 100%

重点函数：
- loss_l2: 简单函数，目标100%
- loss_distil: 简单函数，目标100%
- loss_distil_p: 简单函数，目标100%
- loss_distil_pixel: 复杂函数，目标95%+
- calculate_pixel_similarity: 复杂函数，目标95%+

### eval/metrics_engineering.py
- **语句覆盖**: 95%+
- **分支覆盖**: 100%
- **函数覆盖**: 100%

重点函数：
- count_trainable_params: 简单条件，目标100%
- measure_inference_latency: 多分支，目标100%
- measure_peak_gpu_memory: 简单分支，目标100%
- compute_fp_per_image: 多条件，目标100%

## 测试环境要求

### 基础环境
- Python 3.8+
- PyTorch 2.0+
- pytest 7.4+
- pytest-cov 4.1+

### 可选环境
- CUDA 设备（用于 CUDA 分支测试）
- GPU 显存 >= 2GB

## 执行测试命令

```bash
# 运行所有白盒测试
pytest tests/manual/phase1/whitebox/ -v

# 运行特定测试文件
pytest tests/manual/phase1/whitebox/test_statement_coverage.py -v

# 生成覆盖率报告
pytest tests/manual/phase1/whitebox/ --cov=utils.losses --cov=eval.metrics_engineering --cov-report=html -v

# 查看覆盖率报告
open tests/manual/phase1/coverage_reports/index.html

# 运行特定标记的测试
pytest tests/manual/phase1/whitebox/ -m "whitebox and phase1" -v

# 生成详细的覆盖率报告
pytest tests/manual/phase1/whitebox/ \
    --cov=utils.losses \
    --cov=eval.metrics_engineering \
    --cov-report=html:tests/manual/phase1/coverage_reports \
    --cov-report=term-missing \
    -v

# 只测试CPU相关功能（无CUDA环境）
pytest tests/manual/phase1/whitebox/ -v -k "not cuda"
```

## 覆盖率分析

### 未覆盖代码说明

#### 1. CUDA相关分支
- **位置**: `measure_inference_latency`, `measure_peak_gpu_memory`
- **原因**: 需要CUDA环境
- **覆盖率影响**: 约5%
- **建议**: CI环境配置GPU runner

#### 2. 注释掉的历史代码
- **位置**: `utils/losses.py` line 179-251
- **原因**: 实验性代码，未启用
- **覆盖率影响**: 0% (已注释)
- **建议**: 保持不覆盖

#### 3. 异常处理路径
- **位置**: 各函数的异常分支
- **原因**: 需要构造异常场景
- **覆盖率影响**: 约2%
- **建议**: 增加异常测试用例

### 圈复杂度分析

| 函数名 | 圈复杂度 | 独立路径数 | 测试用例数 | 覆盖状态 |
|--------|---------|-----------|-----------|---------|
| loss_l2 | 2 | 2 | 3 | 充分 |
| loss_distil | 2 | 2 | 2 | 充分 |
| loss_distil_p | 2 | 2 | 1 | 充分 |
| loss_distil_pixel | 4 | 4 | 4 | 充分 |
| calculate_pixel_similarity | 3 | 3 | 2 | 充分 |
| count_trainable_params | 2 | 2 | 3 | 充分 |
| measure_inference_latency | 3 | 3 | 3 | 充分 |
| measure_peak_gpu_memory | 2 | 2 | 2 | 充分 |
| compute_fp_per_image | 2 | 2 | 4 | 充分 |

## 已发现缺陷（白盒测试）

### 缺陷 DEF-WB-001: 空列表未单独处理
- **严重程度**: 低
- **位置**: loss_l2, loss_distil, loss_distil_p
- **描述**: 空特征列表时直接返回初始值0.0，缺少显式检查
- **建议**: 添加长度检查并记录警告日志

### 缺陷 DEF-WB-002: CUDA同步开销未优化
- **严重程度**: 低
- **位置**: measure_inference_latency
- **描述**: 每次warmup后都同步，可能影响性能测试准确性
- **建议**: 仅在关键点同步

### 缺陷 DEF-WB-003: 除零保护不完整
- **严重程度**: 中
- **位置**: count_trainable_params
- **描述**: 虽有 total > 0 检查，但 trainable 可能为0导致误解
- **建议**: 分别检查 total 和 trainable

## 覆盖率度量标准

### 优秀 (90-100%)
- 语句覆盖: ≥95%
- 分支覆盖: ≥95%
- 条件覆盖: ≥90%
- 路径覆盖: ≥85%

### 良好 (80-89%)
- 语句覆盖: 85-94%
- 分支覆盖: 85-94%
- 条件覆盖: 80-89%
- 路径覆盖: 75-84%

### 及格 (70-79%)
- 语句覆盖: 75-84%
- 分支覆盖: 75-84%
- 条件覆盖: 70-79%
- 路径覆盖: 65-74%

### 当前预期
- **语句覆盖**: 95%+ (优秀)
- **分支覆盖**: 100% (优秀)
- **条件覆盖**: 100% (优秀)
- **路径覆盖**: 90%+ (优秀)

## 下一步

1. ✅ 完成所有白盒测试用例编写
2. ⏳ 执行测试并生成覆盖率报告
3. ⏳ 分析未覆盖代码并补充测试
4. ⏳ 修复发现的缺陷
5. ⏳ 更新测试文档
6. ⏳ 开始模块二测试（自动化、性能、安全、集成）

## 测试用例复用性

这些白盒测试用例可用于：
- 回归测试
- 持续集成
- 代码重构验证
- 性能基准测试的基础
