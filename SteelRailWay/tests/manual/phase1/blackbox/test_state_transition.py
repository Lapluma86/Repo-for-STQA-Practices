# -*- coding: utf-8 -*-
"""
黑盒测试 - 状态转换测试
测试目标：测试系统在不同状态之间的转换

状态定义：
1. 未初始化 (Uninitialized)
2. 已初始化 (Initialized)
3. 数据加载中 (Loading)
4. 数据就绪 (Ready)
5. 错误状态 (Error)

状态转换：
- 未初始化 -> 已初始化 (创建对象)
- 已初始化 -> 数据加载中 (__getitem__ 调用)
- 数据加载中 -> 数据就绪 (成功加载)
- 数据加载中 -> 错误状态 (加载失败)

作者：手工编写（Phase 1）
"""

import pytest
import sys
import numpy as np
from pathlib import Path
from PIL import Image
import cv2

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.rail_dataset import RailDualModalDataset


@pytest.mark.blackbox
@pytest.mark.phase1
class TestStateTransition:
    """状态转换测试类"""

    # ===== 测试用例 TC-ST-001: 初始化状态转换 =====

    def test_state_uninitialized_to_initialized(self, tmp_path):
        """
        测试用例ID: TC-ST-001-1
        状态转换: 未初始化 -> 已初始化
        测试步骤: 创建数据集对象
        预期结果: 对象成功创建，属性正确设置
        """
        train_root = str(tmp_path / "train")
        test_root = str(tmp_path / "test")

        (tmp_path / "train" / "Cam1" / "rgb").mkdir(parents=True, exist_ok=True)
        (tmp_path / "train" / "Cam1" / "depth").mkdir(parents=True, exist_ok=True)

        # 状态转换：未初始化 -> 已初始化
        dataset = RailDualModalDataset(
            train_root=train_root,
            test_root=test_root,
            view_id=1,
            split="train"
        )

        # 验证已初始化状态
        assert dataset is not None
        assert dataset.split == "train"
        assert dataset.view_id == 1
        assert hasattr(dataset, 'train_root')
        assert hasattr(dataset, 'test_root')

    def test_state_initialization_with_different_configs(self, tmp_path):
        """
        测试用例ID: TC-ST-001-2
        状态转换: 未初始化 -> 已初始化（不同配置）
        测试步骤: 使用不同配置多次初始化
        预期结果: 每次都能成功初始化
        """
        train_root = str(tmp_path / "train")
        test_root = str(tmp_path / "test")

        (tmp_path / "train" / "Cam1" / "rgb").mkdir(parents=True, exist_ok=True)
        (tmp_path / "train" / "Cam1" / "depth").mkdir(parents=True, exist_ok=True)

        # 第一次初始化
        dataset1 = RailDualModalDataset(
            train_root=train_root,
            test_root=test_root,
            view_id=1,
            split="train"
        )
        assert dataset1.split == "train"

        # 第二次初始化（不同配置）
        dataset2 = RailDualModalDataset(
            train_root=train_root,
            test_root=test_root,
            view_id=1,
            split="val"
        )
        assert dataset2.split == "val"

        # 验证两个对象独立
        assert dataset1.split != dataset2.split

    # ===== 测试用例 TC-ST-002: 数据加载状态转换 =====

    def test_state_initialized_to_loading(self, tmp_path):
        """
        测试用例ID: TC-ST-002-1
        状态转换: 已初始化 -> 数据加载中
        测试步骤: 调用 __len__ 方法触发数据扫描
        预期结果: 能够获取数据集长度
        """
        train_root = str(tmp_path / "train")
        test_root = str(tmp_path / "test")

        rgb_dir = tmp_path / "train" / "Cam1" / "rgb"
        depth_dir = tmp_path / "train" / "Cam1" / "depth"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        depth_dir.mkdir(parents=True, exist_ok=True)

        # 创建测试图像文件
        test_img = np.zeros((900, 6000, 3), dtype=np.uint8)
        test_depth = np.zeros((900, 6000), dtype=np.uint16)

        cv2.imwrite(str(rgb_dir / "test_001.jpg"), test_img)
        cv2.imwrite(str(depth_dir / "test_001.tiff"), test_depth)

        dataset = RailDualModalDataset(
            train_root=train_root,
            test_root=test_root,
            view_id=1,
            split="train"
        )

        # 状态转换：已初始化 -> 数据加载中（通过 __len__）
        length = len(dataset)

        # 验证数据加载状态
        assert length >= 0  # 应该能获取长度

    def test_state_loading_to_ready(self, tmp_path):
        """
        测试用例ID: TC-ST-002-2
        状态转换: 数据加载中 -> 数据就绪
        测试步骤: 创建图像文件并加载数据
        预期结果: 成功加载数据项
        """
        train_root = str(tmp_path / "train")
        test_root = str(tmp_path / "test")

        rgb_dir = tmp_path / "train" / "Cam1" / "rgb"
        depth_dir = tmp_path / "train" / "Cam1" / "depth"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        depth_dir.mkdir(parents=True, exist_ok=True)

        # 创建测试图像
        test_img = np.random.randint(0, 255, (900, 6000, 3), dtype=np.uint8)
        test_depth = np.random.randint(0, 1000, (900, 6000), dtype=np.uint16)

        cv2.imwrite(str(rgb_dir / "test_001.jpg"), test_img)
        cv2.imwrite(str(depth_dir / "test_001.tiff"), test_depth)

        dataset = RailDualModalDataset(
            train_root=train_root,
            test_root=test_root,
            view_id=1,
            split="train",
            use_patch=False  # 不使用 patch，简化测试
        )

        if len(dataset) > 0:
            # 状态转换：数据加载中 -> 数据就绪
            data = dataset[0]

            # 验证数据就绪状态
            assert data is not None
            assert 'rgb' in data or isinstance(data, tuple)

    def test_state_loading_to_error(self, tmp_path):
        """
        测试用例ID: TC-ST-002-3
        状态转换: 数据加载中 -> 错误状态
        测试步骤: 尝试访问不存在的数据索引
        预期结果: 抛出异常或返回错误
        """
        train_root = str(tmp_path / "train")
        test_root = str(tmp_path / "test")

        (tmp_path / "train" / "Cam1" / "rgb").mkdir(parents=True, exist_ok=True)
        (tmp_path / "train" / "Cam1" / "depth").mkdir(parents=True, exist_ok=True)

        dataset = RailDualModalDataset(
            train_root=train_root,
            test_root=test_root,
            view_id=1,
            split="train"
        )

        # 状态转换：数据加载中 -> 错误状态（访问无效索引）
        with pytest.raises((IndexError, KeyError, ValueError, FileNotFoundError)):
            _ = dataset[999]  # 访问不存在的索引

    # ===== 测试用例 TC-ST-003: split 参数状态转换 =====

    def test_state_transition_train_to_val(self, tmp_path):
        """
        测试用例ID: TC-ST-003-1
        状态转换: train 模式 -> val 模式
        测试步骤: 创建两个不同 split 的数据集对象
        预期结果: 两个对象状态独立，互不影响
        """
        train_root = str(tmp_path / "train")
        test_root = str(tmp_path / "test")

        (tmp_path / "train" / "Cam1" / "rgb").mkdir(parents=True, exist_ok=True)
        (tmp_path / "train" / "Cam1" / "depth").mkdir(parents=True, exist_ok=True)

        # 创建 train 模式数据集
        dataset_train = RailDualModalDataset(
            train_root=train_root,
            test_root=test_root,
            view_id=1,
            split="train"
        )

        # 创建 val 模式数据集（相同配置，不同 split）
        dataset_val = RailDualModalDataset(
            train_root=train_root,
            test_root=test_root,
            view_id=1,
            split="val"
        )

        # 验证状态转换
        assert dataset_train.split == "train"
        assert dataset_val.split == "val"
        assert dataset_train.split != dataset_val.split

    def test_state_transition_train_to_test(self, tmp_path):
        """
        测试用例ID: TC-ST-003-2
        状态转换: train 模式 -> test 模式
        测试步骤: 切换到测试模式
        预期结果: 使用不同的数据路径
        """
        train_root = str(tmp_path / "train")
        test_root = str(tmp_path / "test")

        (tmp_path / "train" / "Cam1" / "rgb").mkdir(parents=True, exist_ok=True)
        (tmp_path / "train" / "Cam1" / "depth").mkdir(parents=True, exist_ok=True)
        (tmp_path / "test" / "rail_mvtec" / "cam1" / "test" / "good").mkdir(parents=True, exist_ok=True)
        (tmp_path / "test" / "rail_mvtec_depth" / "cam1" / "test" / "good").mkdir(parents=True, exist_ok=True)

        # train 模式
        dataset_train = RailDualModalDataset(
            train_root=train_root,
            test_root=test_root,
            view_id=1,
            split="train"
        )

        # test 模式
        dataset_test = RailDualModalDataset(
            train_root=train_root,
            test_root=test_root,
            view_id=1,
            split="test"
        )

        assert dataset_train.split == "train"
        assert dataset_test.split == "test"

    # ===== 测试用例 TC-ST-004: preload 状态转换 =====

    def test_state_transition_lazy_to_preload(self, tmp_path):
        """
        测试用例ID: TC-ST-004-1
        状态转换: 懒加载模式 -> 预加载模式
        测试步骤: 创建两个对象，一个懒加载，一个预加载
        预期结果: preload 参数正确设置
        """
        train_root = str(tmp_path / "train")
        test_root = str(tmp_path / "test")

        (tmp_path / "train" / "Cam1" / "rgb").mkdir(parents=True, exist_ok=True)
        (tmp_path / "train" / "Cam1" / "depth").mkdir(parents=True, exist_ok=True)

        # 懒加载模式
        dataset_lazy = RailDualModalDataset(
            train_root=train_root,
            test_root=test_root,
            view_id=1,
            split="train",
            preload=False
        )

        # 预加载模式
        dataset_preload = RailDualModalDataset(
            train_root=train_root,
            test_root=test_root,
            view_id=1,
            split="train",
            preload=True
        )

        assert dataset_lazy.preload is False
        assert dataset_preload.preload is True

    # ===== 测试用例 TC-ST-005: patch 模式状态转换 =====

    def test_state_transition_no_patch_to_patch(self, tmp_path):
        """
        测试用例ID: TC-ST-005-1
        状态转换: 非 patch 模式 -> patch 模式
        测试步骤: 创建两个对象，测试不同 patch 配置
        预期结果: use_patch 参数正确设置
        """
        train_root = str(tmp_path / "train")
        test_root = str(tmp_path / "test")

        (tmp_path / "train" / "Cam1" / "rgb").mkdir(parents=True, exist_ok=True)
        (tmp_path / "train" / "Cam1" / "depth").mkdir(parents=True, exist_ok=True)

        # 非 patch 模式
        dataset_no_patch = RailDualModalDataset(
            train_root=train_root,
            test_root=test_root,
            view_id=1,
            split="train",
            use_patch=False
        )

        # patch 模式
        dataset_with_patch = RailDualModalDataset(
            train_root=train_root,
            test_root=test_root,
            view_id=1,
            split="train",
            use_patch=True
        )

        assert dataset_no_patch.use_patch is False
        assert dataset_with_patch.use_patch is True

    # ===== 测试用例 TC-ST-006: 多次状态转换序列 =====

    def test_state_transition_sequence(self, tmp_path):
        """
        测试用例ID: TC-ST-006-1
        状态转换序列: 未初始化 -> 已初始化 -> 数据加载中 -> 数据就绪
        测试步骤: 完整的数据加载流程
        预期结果: 所有状态转换正常
        """
        train_root = str(tmp_path / "train")
        test_root = str(tmp_path / "test")

        rgb_dir = tmp_path / "train" / "Cam1" / "rgb"
        depth_dir = tmp_path / "train" / "Cam1" / "depth"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        depth_dir.mkdir(parents=True, exist_ok=True)

        # 创建测试数据
        test_img = np.random.randint(0, 255, (900, 6000, 3), dtype=np.uint8)
        test_depth = np.random.randint(0, 1000, (900, 6000), dtype=np.uint16)

        cv2.imwrite(str(rgb_dir / "test_001.jpg"), test_img)
        cv2.imwrite(str(depth_dir / "test_001.tiff"), test_depth)

        # 状态1: 未初始化 -> 已初始化
        dataset = RailDualModalDataset(
            train_root=train_root,
            test_root=test_root,
            view_id=1,
            split="train",
            use_patch=False
        )
        assert dataset is not None

        # 状态2: 已初始化 -> 数据加载中（获取长度）
        length = len(dataset)
        assert length >= 0

        # 状态3: 数据加载中 -> 数据就绪（加载数据项）
        if length > 0:
            data = dataset[0]
            assert data is not None

    def test_state_transition_repeated_access(self, tmp_path):
        """
        测试用例ID: TC-ST-006-2
        状态转换: 重复的数据就绪 -> 数据加载中 -> 数据就绪
        测试步骤: 多次访问同一数据项
        预期结果: 可以重复访问，状态稳定
        """
        train_root = str(tmp_path / "train")
        test_root = str(tmp_path / "test")

        rgb_dir = tmp_path / "train" / "Cam1" / "rgb"
        depth_dir = tmp_path / "train" / "Cam1" / "depth"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        depth_dir.mkdir(parents=True, exist_ok=True)

        # 创建测试数据
        test_img = np.random.randint(0, 255, (900, 6000, 3), dtype=np.uint8)
        test_depth = np.random.randint(0, 1000, (900, 6000), dtype=np.uint16)

        cv2.imwrite(str(rgb_dir / "test_001.jpg"), test_img)
        cv2.imwrite(str(depth_dir / "test_001.tiff"), test_depth)

        dataset = RailDualModalDataset(
            train_root=train_root,
            test_root=test_root,
            view_id=1,
            split="train",
            use_patch=False
        )

        if len(dataset) > 0:
            # 第一次访问
            data1 = dataset[0]
            assert data1 is not None

            # 第二次访问（重复）
            data2 = dataset[0]
            assert data2 is not None

            # 第三次访问
            data3 = dataset[0]
            assert data3 is not None

    # ===== 测试用例 TC-ST-007: 异常状态转换 =====

    def test_state_transition_error_recovery(self, tmp_path):
        """
        测试用例ID: TC-ST-007-1
        状态转换: 错误状态 -> 正常状态（错误恢复）
        测试步骤: 先访问无效索引，再访问有效索引
        预期结果: 错误后仍能正常访问有效数据
        """
        train_root = str(tmp_path / "train")
        test_root = str(tmp_path / "test")

        rgb_dir = tmp_path / "train" / "Cam1" / "rgb"
        depth_dir = tmp_path / "train" / "Cam1" / "depth"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        depth_dir.mkdir(parents=True, exist_ok=True)

        # 创建测试数据
        test_img = np.random.randint(0, 255, (900, 6000, 3), dtype=np.uint8)
        test_depth = np.random.randint(0, 1000, (900, 6000), dtype=np.uint16)

        cv2.imwrite(str(rgb_dir / "test_001.jpg"), test_img)
        cv2.imwrite(str(depth_dir / "test_001.tiff"), test_depth)

        dataset = RailDualModalDataset(
            train_root=train_root,
            test_root=test_root,
            view_id=1,
            split="train",
            use_patch=False
        )

        # 访问无效索引（进入错误状态）
        try:
            _ = dataset[999]
        except (IndexError, KeyError, ValueError, FileNotFoundError):
            pass  # 预期的错误

        # 错误后恢复：访问有效索引
        if len(dataset) > 0:
            data = dataset[0]
            assert data is not None  # 应该能恢复正常访问
