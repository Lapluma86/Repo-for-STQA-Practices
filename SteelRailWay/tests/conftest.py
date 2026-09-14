# -*- coding: utf-8 -*-
"""
pytest 配置文件
定义共享的 fixtures 和测试配置
"""

import pytest
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def project_root():
    """项目根目录"""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def test_data_dir():
    """测试数据目录"""
    return PROJECT_ROOT / "datasets"


@pytest.fixture(scope="session")
def config_dir():
    """配置文件目录"""
    return PROJECT_ROOT / "configs"


@pytest.fixture(scope="session")
def output_dir(tmp_path_factory):
    """临时输出目录"""
    return tmp_path_factory.mktemp("test_outputs")


@pytest.fixture
def sample_image_path():
    """示例图像路径（需要根据实际情况调整）"""
    # 这里返回一个示例路径，实际测试时需要有真实图像
    return PROJECT_ROOT / "datasets" / "sample.png"


@pytest.fixture
def device():
    """测试使用的设备"""
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(autouse=True)
def reset_random_seed():
    """每个测试前重置随机种子，确保可重复性"""
    import random
    import numpy as np
    import torch

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pytest_configure(config):
    """pytest 配置钩子"""
    # 添加自定义标记
    config.addinivalue_line(
        "markers", "blackbox: 黑盒测试标记"
    )
    config.addinivalue_line(
        "markers", "whitebox: 白盒测试标记"
    )
    config.addinivalue_line(
        "markers", "performance: 性能测试标记"
    )
    config.addinivalue_line(
        "markers", "security: 安全测试标记"
    )
    config.addinivalue_line(
        "markers", "integration: 集成测试标记"
    )
    config.addinivalue_line(
        "markers", "slow: 运行时间较长的测试"
    )
    config.addinivalue_line(
        "markers", "phase1: 第一阶段测试（人工手写）"
    )
    config.addinivalue_line(
        "markers", "phase2: 第二阶段测试（AI辅助）"
    )
