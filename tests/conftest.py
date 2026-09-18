# -*- coding: utf-8 -*-
"""
pytest 配置文件
定义共享的 fixtures 和测试配置
"""

import pytest
import sys
from pathlib import Path

# 添加项目根目录到路径
REPO_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = REPO_ROOT / "SteelRailWay"
sys.path.insert(0, str(PROJECT_ROOT))


def pytest_sessionstart(session):
    """冻结被测源码：收集前核对清单，防止修复混入缺陷演示。"""
    import hashlib
    import json

    manifest = json.loads((REPO_ROOT / "tests/source-freeze.json").read_text(encoding="utf-8"))
    changed = []
    for relative, expected in manifest["sha256"].items():
        path = REPO_ROOT / relative
        if not path.is_file():
            changed.append(relative)
            continue
        data = path.read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(data).hexdigest() != expected:
            changed.append(relative)
    if changed:
        raise pytest.UsageError(
            "被测源码偏离冻结基线；请核对修改，勿为测试通过更新指纹：\n"
            + "\n".join(changed)
        )


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
    return torch.device("cpu")


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



@pytest.fixture
def rail_factory(tmp_path):
    """小型真实双模态数据；显式划分使加载用例不会意外得到空训练集。"""
    import cv2
    import numpy as np
    from datasets.rail_dataset import RailDualModalDataset

    def create(count=1, **kwargs):
        root = tmp_path / "train"
        rgb_dir, depth_dir = root / "Cam1/rgb", root / "Cam1/depth"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        depth_dir.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            rgb = np.full((16, 24, 3), 40 + i, dtype=np.uint8)
            depth = np.arange(1, 385, dtype=np.uint16).reshape(16, 24) + i
            assert cv2.imwrite(str(rgb_dir / f"frame_{i:03}.jpg"), rgb)
            assert cv2.imwrite(str(depth_dir / f"frame_{i:03}.tiff"), depth)
        options = dict(train_root=str(root), test_root=str(tmp_path / "test"),
                       view_id=1, split="train", img_size=8, use_patch=False,
                       patch_size=8, patch_stride=8, preload_workers=1,
                       train_val_test_split=[1, 0, 0])
        options.update(kwargs)
        return RailDualModalDataset(**options)
    return create
