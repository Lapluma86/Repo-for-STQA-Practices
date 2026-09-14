# SteelRailWay 测试框架

## 目录结构

```
tests/
├── manual/              # 模块一：手工测试
│   ├── phase1/         # 第一阶段：手工测试用例
│   │   ├── blackbox/   # 黑盒测试
│   │   └── whitebox/   # 白盒测试
│   └── phase2/         # 第二阶段：AI辅助测试
├── automated/           # 模块二：自动化测试
├── performance/         # 模块二：性能测试
├── security/            # 模块二：安全测试
└── integration/         # 模块二：集成测试
```

## 测试策略

### 模块一：基础测试

**第一阶段 - 手工测试用例**

黑盒测试：
- 等价类划分
- 边界值分析
- 判定表测试
- 状态转换测试

白盒测试：
- 语句覆盖
- 分支覆盖
- 条件覆盖
- 路径覆盖

**第二阶段 - AI辅助测试**
- AI生成的补充测试用例
- 覆盖率优化
- 手工测试与AI测试对比

### 模块二：高级测试

- 使用pytest的自动化测试
- 性能测试（推理速度、内存占用）
- 安全测试（输入验证、路径安全）
- 集成测试（端到端工作流）

## 运行测试

```bash
# 运行所有测试
pytest tests/

# 运行特定模块
pytest tests/manual/phase1/blackbox/

# 生成覆盖率报告
pytest --cov=. --cov-report=html tests/
```
