# 黑盒测试用例清单 - 模块一第一阶段

## 测试概览
- **项目名称**: SteelRailWay 铁路缺陷检测系统
- **测试阶段**: 模块一 - 第一阶段（人工手写）
- **测试类型**: 黑盒测试
- **测试日期**: 2024年
- **测试人员**: 手工编写

## 测试统计

### 等价类划分测试
- **测试文件**: test_equivalence_partitioning.py
- **测试用例数**: 25个
- **覆盖参数**: view_id, split, img_size, depth_norm, train_sample_ratio

| 测试用例ID | 测试目的 | 等价类类型 | 预期结果 | 实际结果 | 状态 |
|-----------|---------|-----------|---------|---------|------|
| TC-EP-001-1 | view_id 有效范围 | 有效等价类 | 正常创建 | - | 待测试 |
| TC-EP-001-2 | view_id 下边界 | 边界等价类 | 正常创建 | - | 待测试 |
| TC-EP-001-3 | view_id 上边界 | 边界等价类 | 正常创建 | - | 待测试 |
| TC-EP-001-4 | view_id=0 | 无效等价类 | 拒绝/异常 | 接受（缺陷） | 待测试 |
| TC-EP-001-5 | view_id 负数 | 无效等价类 | 拒绝/异常 | 接受（缺陷） | 待测试 |
| TC-EP-002-1 | split='train' | 有效等价类 | 正常创建 | - | 待测试 |
| TC-EP-002-2 | split='val' | 有效等价类 | 正常创建 | - | 待测试 |
| TC-EP-002-3 | split='test' | 有效等价类 | 正常创建 | - | 待测试 |
| TC-EP-002-4 | split='' | 无效等价类 | 拒绝/异常 | 接受（缺陷） | 待测试 |
| TC-EP-002-5 | split='unknown' | 无效等价类 | 拒绝/异常 | 接受（缺陷） | 待测试 |

### 边界值分析测试
- **测试文件**: test_boundary_value.py
- **测试用例数**: 30个
- **覆盖参数**: img_size, patch_size, patch_stride, train_sample_ratio, view_id, 路径参数

| 测试用例ID | 测试目的 | 边界值 | 预期结果 | 实际结果 | 状态 |
|-----------|---------|--------|---------|---------|------|
| TC-BV-001-1 | img_size 最小值 | 1 | 正常创建 | - | 待测试 |
| TC-BV-001-2 | img_size 最小值+1 | 2 | 正常创建 | - | 待测试 |
| TC-BV-001-3 | img_size 常用值 | 224/256/512/1024 | 正常创建 | - | 待测试 |
| TC-BV-001-4 | img_size 超大值 | 10000 | 正常创建 | - | 待测试 |
| TC-BV-001-5 | img_size 零值 | 0 | 拒绝/异常 | 接受（缺陷） | 待测试 |

### 判定表测试
- **测试文件**: test_decision_table.py
- **测试用例数**: 15个
- **测试条件**: use_patch, preload, split, depth_norm 的组合

| 测试用例ID | 条件组合 | 预期结果 | 实际结果 | 状态 |
|-----------|---------|---------|---------|------|
| TC-DT-001-1 | patch=T, split=train | 正常工作 | - | 待测试 |
| TC-DT-001-2 | patch=F, split=train | 正常工作 | - | 待测试 |
| TC-DT-001-3 | patch=T, split=test | 正常工作 | - | 待测试 |
| TC-DT-001-4 | patch=F, split=test | 正常工作 | - | 待测试 |

### 状态转换测试
- **测试文件**: test_state_transition.py
- **测试用例数**: 15个
- **状态**: 未初始化, 已初始化, 数据加载中, 数据就绪, 错误状态

| 测试用例ID | 状态转换 | 预期结果 | 实际结果 | 状态 |
|-----------|---------|---------|---------|------|
| TC-ST-001-1 | 未初始化->已初始化 | 成功创建对象 | - | 待测试 |
| TC-ST-002-1 | 已初始化->数据加载中 | 获取数据集长度 | - | 待测试 |
| TC-ST-002-2 | 数据加载中->数据就绪 | 成功加载数据 | - | 待测试 |
| TC-ST-002-3 | 数据加载中->错误状态 | 抛出异常 | - | 待测试 |

## 已发现缺陷汇总

### 缺陷 DEF-001: 输入参数缺乏验证
- **严重程度**: 中
- **影响模块**: RailDualModalDataset.__init__
- **描述**: 
  - view_id 允许 0 和负数
  - split 允许任意字符串
  - img_size 允许 0 和负数
  - train_sample_ratio 允许负数和大于 1 的值
- **建议**: 添加参数验证逻辑

### 缺陷 DEF-002: 路径参数未验证
- **严重程度**: 低
- **影响模块**: RailDualModalDataset.__init__
- **描述**: 允许空字符串和不存在的路径
- **建议**: 在初始化时验证路径有效性

### 缺陷 DEF-003: depth_norm 参数未验证
- **严重程度**: 中
- **影响模块**: RailDualModalDataset.__init__
- **描述**: 接受无效的归一化方法名称
- **建议**: 限制为 ['zscore', 'minmax', 'log']

### 缺陷 DEF-004: patch_stride=0 未验证
- **严重程度**: 高
- **影响模块**: RailDualModalDataset.patch 逻辑
- **描述**: patch_stride=0 可能导致无限循环
- **建议**: 确保 patch_stride > 0

## 测试覆盖率目标
- [ ] 参数等价类覆盖率: 100%
- [ ] 边界值覆盖率: 100%
- [ ] 条件组合覆盖率: 80%+
- [ ] 状态转换覆盖率: 100%

## 执行测试命令

```bash
# 运行所有黑盒测试
pytest tests/manual/phase1/blackbox/ -v

# 运行特定测试文件
pytest tests/manual/phase1/blackbox/test_equivalence_partitioning.py -v

# 运行带标记的测试
pytest tests/manual/phase1/blackbox/ -m "blackbox and phase1" -v

# 生成详细报告
pytest tests/manual/phase1/blackbox/ -v --tb=long > blackbox_test_report.txt
```

## 下一步
1. 执行所有黑盒测试用例
2. 记录实际测试结果
3. 补充缺陷报告详情
4. 进行白盒测试
