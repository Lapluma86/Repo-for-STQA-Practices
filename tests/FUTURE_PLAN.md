> 历史文档：本文保留迁移前的阶段名称、命令和统计，可能已过期。当前目录、运行方式及课程要求以仓库根目录 `tests/README.md` 为准；原 `manual/phase1` 已迁至 `module1`，原 `manual/phase2` 已迁至 `module2`。

# SteelRailWay 测试 - 未来计划

## 当前状态

### 已完成
- 已编写 170 个测试用例（黑盒测试 85 个 + 白盒测试 85 个）
- 建立测试基础设施
- 发现并修复 12 个缺陷
- 完成测试文档

### 待完成
- 测试执行
- 生成覆盖率报告

---

## 推荐计划

### 第一阶段：验证（第1周）

#### 1.1 执行测试

```bash
pip install -r tests/requirements-test.txt
pytest tests/manual/phase1/ -v
```

#### 1.2 修复后更新测试

- 使测试用例与新的异常行为保持一致
- 为参数验证添加正向测试
- 添加边界值测试

---

### 第二阶段：模块一第二阶段 - AI辅助测试（第2周）

#### 2.1 AI生成的测试

- 使用AI生成额外的边界情况
- 对比AI生成的测试与手工测试的覆盖率
- 分析缺失的场景

#### 2.2 测试优化

- 合并冗余测试
- 添加参数化测试

---

### 第三阶段：模块二 - 高级测试（第3周）

#### 3.1 自动化测试
```
tests/automated/
├── test_data_loading.py
├── test_model_init.py
├── test_training.py
└── test_inference.py
```

#### 3.2 性能测试
```
tests/performance/
├── test_inference_speed.py
├── test_memory_usage.py
├── test_gpu_utilization.py
└── test_scaling.py
```

#### 3.3 安全测试
```
tests/security/
├── test_input_validation.py
├── test_path_traversal.py
└── test_data_integrity.py
```

#### 3.4 集成测试
```
tests/integration/
├── test_end_to_end.py
├── test_multi_view.py
└── test_pretrain_finetune.py
```

---

### 第四阶段：CI/CD 集成（第4周）

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - run: pytest tests/ -v
      - run: pytest --cov=. --cov-report=xml
```

---

## 预计时间表

| 阶段 | 任务 | 持续时间 |
|-------|------|----------|
| 第一阶段 | 验证 | 2-3 天 |
| 第二阶段 | AI辅助测试 | 2-3 天 |
| 第三阶段 | 高级测试 | 3-4 天 |
| 第四阶段 | CI/CD | 1-2 天 |
| **总计** | | **8-12 天** |

---

## 学习目标

1. 掌握 pytest：fixtures、parametrize、markers
2. 测试策略：黑盒测试、白盒测试、灰盒测试
3. AI辅助测试
4. CI/CD 实践
5. 性能测试
