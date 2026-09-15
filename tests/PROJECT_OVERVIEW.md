> 历史文档：本文保留迁移前的阶段名称、命令和统计，可能已过期。当前目录、运行方式及课程要求以仓库根目录 `tests/README.md` 为准；原 `manual/phase1` 已迁至 `module1`，原 `manual/phase2` 已迁至 `module2`。

# SteelRailWay 测试项目概览

## 项目目标

为 SteelRailWay 铁路缺陷检测系统建立完整的测试体系：

- 模块一：基础测试（黑盒测试 + 白盒测试）
- 模块二：高级测试（自动化 + 性能 + 安全 + 集成测试）

## 当前进度

### 模块一 - 第一阶段：手工测试

**状态：已完成**

| 类别 | 用例数 | 结果 |
|----------|-------|--------|
| 黑盒测试 | 88 | 通过 |
| 白盒测试 | 84 | 通过 |
| **总计** | **172** | **161个通过, 11个跳过** |

### 模块一 - 第二阶段：AI辅助测试

**状态：已完成**

| 类别 | 用例数 | 结果 |
|----------|-------|--------|
| Hypothesis 属性测试 | 3 | 通过 |
| 回归测试 | 4 | 通过 |
| **总计** | **7** | **7个通过** |

### 模块二：高级测试

**状态：待开发**

- 自动化测试
- 性能测试
- 安全测试
- 集成测试

## 测试结果

```
总计：179 个测试用例
通过：168 个
跳过：11 个（非GPU环境下的CUDA测试）
耗时：5.34秒
```

## 覆盖率

| 模块 | 覆盖率 |
|--------|----------|
| `utils/losses.py` | 97% |
| `eval/metrics_engineering.py` | 92% |
| **总体** | **96%** |

## 快速开始

```bash
cd SteelRailWay

# 安装依赖
pip install -r tests/requirements-test.txt

# 运行所有测试
pytest tests/ -v

# 生成覆盖率报告
pytest tests/manual/phase1/whitebox/ \
    --cov=utils.losses \
    --cov=eval.metrics_engineering \
    --cov-report=html:tests/manual/phase1/coverage_reports -v
```

## 项目结构

```
tests/
├── manual/phase1/           # 第一阶段已完成
│   ├── blackbox/            # 88个用例
│   └── whitebox/            # 84个用例
├── manual/phase2/           # 第二阶段已完成
│   └── ai_assisted/         # 7个用例
├── automated/               # 模块二（待开发）
├── performance/             # 模块二（待开发）
├── security/               # 模块二（待开发）
├── integration/             # 模块二（待开发）
├── pytest.ini
├── conftest.py
└── requirements-test.txt
```

## 已完成工作

- 已编写 179 个测试用例
- 发现 12 个缺陷
- 修复 7 个缺陷
- 添加参数验证
- 生成覆盖率报告
- 更新文档
