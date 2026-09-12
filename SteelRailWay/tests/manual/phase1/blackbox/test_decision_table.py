# -*- coding: utf-8 -*-
"""
黑盒测试 - 判定表测试
测试目标：测试多个配置参数组合的行为

判定表设计：
条件1: use_patch (True/False)
条件2: preload (True/False)
条件3: split (train/test)
条件4: depth_norm (zscore/minmax/log)

测试不同条件组合下系统的行为

作者：手工编写（Phase 1）
"""

import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.rail_dataset import RailDualModalDataset


@pytest.mark.blackbox
@pytest.mark.phase1
class TestDecisionTable:
    """判定表测试类"""

    # ===== 测试用例 TC-DT-001: use_patch 与 split 组合 =====

    def test_combination_patch_true_split_train(self, tmp_path):
        """
        测试用例ID: TC-DT-001-1
        判定表规则: use_patch=True, split='train'
        预期结果: 正常创建，使用 patch 分割训练数据
        """
        train_root = str(tmp_path / "train")
        test_root = str(tmp_path / "test")

        (tmp_path / "train" / "Cam1" / "rgb").mkdir(parents=True, exist_ok=True)
        (tmp_path / "train" / "Cam1" / "depth").mkdir(parents=True, exist_ok=True)

        dataset = RailDualModalDataset(
            train_root=train_root,
            test_root=test_root,
            view_id=1,
            split="train",
            use_patch=True
        )

        assert dataset.use_patch is True
        assert dataset.split == "train"

    def test_combination_patch_false_split_train(self, tmp_path):
        """
        测试用例ID: TC-DT-001-2
        判定表规则: use_patch=False, split='train'
        预期结果: 正常创建，不使用 patch 分割
        """
        train_root = str(tmp_path / "train")
        test_root = str(tmp_path / "test")

        (tmp_path / "train" / "Cam1" / "rgb").mkdir(parents=True, exist_ok=True)
        (tmp_path / "train" / "Cam1" / "depth").mkdir(parents=True, exist_ok=True)

        dataset = RailDualModalDataset(
            train_root=train_root,
            test_root=test_root,
            view_id=1,
            split="train",
            use_patch=False
        )

        assert dataset.use_patch is False
        assert dataset.split == "train"

    def test_combination_patch_true_split_test(self, tmp_path):
        """
        测试用例ID: TC-DT-001-3
        判定表规则: use_patch=True, split='test'
        预期结果: 正常创建，测试集使用 patch 分割
        """
        train_root = str(tmp_path / "train")
        test_root = str(tmp_path / "test")

        (tmp_path / "test" / "rail_mvtec" / "cam1" / "test" / "good").mkdir(parents=True, exist_ok=True)
        (tmp_path / "test" / "rail_mvtec_depth" / "cam1" / "test" / "good").mkdir(parents=True, exist_ok=True)

        dataset = RailDualModalDataset(
            train_root=train_root,
            test_root=test_root,
            view_id=1,
            split="test",
            use_patch=True
        )

        assert dataset.use_patch is True
        assert dataset.split == "test"

    def test_combination_patch_false_split_test(self, tmp_path):
        """
        测试用例ID: TC-DT-001-4
        判定表规则: use_patch=False, split='test'
        预期结果: 正常创建，测试集不使用 patch 分割
        """
        train_root = str(tmp_path / "train")
        test_root = str(tmp_path / "test")

        (tmp_path / "test" / "rail_mvtec" / "cam1" / "test" / "good").mkdir(parents=True, exist_ok=True)
        (tmp_path / "test" / "rail_mvtec_depth" / "cam1" / "test" / "good").mkdir(parents=True, exist_ok=True)

        dataset = RailDualModalDataset(
            train_root=train_root,
            test_root=test_root,
            view_id=1,
            split="test",
            use_patch=False
        )

        assert dataset.use_patch is False
        assert dataset.split == "test"

    # ===== 测试用例 TC-DT-002: depth_norm 与 use_patch 组合 =====

    def test_combination_zscore_patch_true(self, tmp_path):
        """
        测试用例ID: TC-DT-002-1
        判定表规则: depth_norm='zscore', use_patch=True
        预期结果: 深度归一化使用 zscore 方法，启用 patch
        """
        train_root = str(tmp_path / "train")
        test_root = str(tmp_path / "test")

        (tmp_path / "train" / "Cam1" / "rgb").mkdir(parents=True, exist_ok=True)
        (tmp_path / "train" / "Cam1" / "depth").mkdir(parents=True, exist_ok=True)

        dataset = RailDualModalDataset(
            train_root=train_root,
            test_root=test_root,
            view_id=1,
            split="train",
            depth_norm="zscore",
            use_patch=True
        )

        assert dataset.depth_norm == "zscore"
        assert dataset.use_patch is True

    def test_combination_minmax_patch_false(self, tmp_path):
        """
        测试用例ID: TC-DT-002-2
        判定表规则: depth_norm='minmax', use_patch=False
        预期结果: 深度归一化使用 minmax 方法，不启用 patch
        """
        train_root = str(tmp_path / "train")
        test_root = str(tmp_path / "test")

        (tmp_path / "train" / "Cam1" / "rgb").mkdir(parents=True, exist_ok=True)
        (tmp_path / "train" / "Cam1" / "depth").mkdir(parents=True, exist_ok=True)

        dataset = RailDualModalDataset(
            train_root=train_root,
            test_root=test_root,
            view_id=1,
            split="train",
            depth_norm="minmax",
            use_patch=False
        )

        assert dataset.depth_norm == "minmax"
        assert dataset.use_patch is False

    def test_combination_log_patch_true(self, tmp_path):
        """
        测试用例ID: TC-DT-002-3
        判定表规则: depth_norm='log', use_patch=True
        预期结果: 深度归一化使用 log 方法，启用 patch
        """
        train_root = str(tmp_path / "train")
        test_root = str(tmp_path / "test")

        (tmp_path / "train" / "Cam1" / "rgb").mkdir(parents=True, exist_ok=True)
        (tmp_path / "train" / "Cam1" / "depth").mkdir(parents=True, exist_ok=True)

        dataset = RailDualModalDataset(
            train_root=train_root,
            test_root=test_root,
            view_id=1,
            split="train",
            depth_norm="log",
            use_patch=True
        )

        assert dataset.depth_norm == "log"
        assert dataset.use_patch is True

    # ===== 测试用例 TC-DT-003: preload 与 split 组合 =====

    def test_combination_preload_true_split_train(self, tmp_path):
        """
        测试用例ID: TC-DT-003-1
        判定表规则: preload=True, split='train'
        预期结果: 预加载训练数据到内存
        """
        train_root = str(tmp_path / "train")
        test_root = str(tmp_path / "test")

        (tmp_path / "train" / "Cam1" / "rgb").mkdir(parents=True, exist_ok=True)
        (tmp_path / "train" / "Cam1" / "depth").mkdir(parents=True, exist_ok=True)

        dataset = RailDualModalDataset(
            train_root=train_root,
            test_root=test_root,
            view_id=1,
            split="train",
            preload=True
        )

        assert dataset.preload is True
        assert dataset.split == "train"

    def test_combination_preload_false_split_test(self, tmp_path):
        """
        测试用例ID: TC-DT-003-2
        判定表规则: preload=False, split='test'
        预期结果: 不预加载测试数据，动态加载
        """
        train_root = str(tmp_path / "train")
        test_root = str(tmp_path / "test")

        (tmp_path / "test" / "rail_mvtec" / "cam1" / "test" / "good").mkdir(parents=True, exist_ok=True)
        (tmp_path / "test" / "rail_mvtec_depth" / "cam1" / "test" / "good").mkdir(parents=True, exist_ok=True)

        dataset = RailDualModalDataset(
            train_root=train_root,
            test_root=test_root,
            view_id=1,
            split="test",
            preload=False
        )

        assert dataset.preload is False
        assert dataset.split == "test"

    # ===== 测试用例 TC-DT-004: 复杂组合测试 =====

    def test_combination_complex_1(self, tmp_path):
        """
        测试用例ID: TC-DT-004-1
        判定表规则: use_patch=True, preload=True, depth_norm='zscore', split='train'
        预期结果: 所有选项组合正常工作
        """
        train_root = str(tmp_path / "train")
        test_root = str(tmp_path / "test")

        (tmp_path / "train" / "Cam1" / "rgb").mkdir(parents=True, exist_ok=True)
        (tmp_path / "train" / "Cam1" / "depth").mkdir(parents=True, exist_ok=True)

        dataset = RailDualModalDataset(
            train_root=train_root,
            test_root=test_root,
            view_id=1,
            split="train",
            use_patch=True,
            preload=True,
            depth_norm="zscore"
        )

        assert dataset.use_patch is True
        assert dataset.preload is True
        assert dataset.depth_norm == "zscore"
        assert dataset.split == "train"

    def test_combination_complex_2(self, tmp_path):
        """
        测试用例ID: TC-DT-004-2
        判定表规则: use_patch=False, preload=False, depth_norm='minmax', split='test'
        预期结果: 最小化配置正常工作
        """
        train_root = str(tmp_path / "train")
        test_root = str(tmp_path / "test")

        (tmp_path / "test" / "rail_mvtec" / "cam1" / "test" / "good").mkdir(parents=True, exist_ok=True)
        (tmp_path / "test" / "rail_mvtec_depth" / "cam1" / "test" / "good").mkdir(parents=True, exist_ok=True)

        dataset = RailDualModalDataset(
            train_root=train_root,
            test_root=test_root,
            view_id=1,
            split="test",
            use_patch=False,
            preload=False,
            depth_norm="minmax"
        )

        assert dataset.use_patch is False
        assert dataset.preload is False
        assert dataset.depth_norm == "minmax"
        assert dataset.split == "test"

    def test_combination_complex_3(self, tmp_path):
        """
        测试用例ID: TC-DT-004-3
        判定表规则: use_patch=True, preload=False, depth_norm='log', split='val'
        预期结果: 验证集配置正常工作
        """
        train_root = str(tmp_path / "train")
        test_root = str(tmp_path / "test")

        (tmp_path / "train" / "Cam1" / "rgb").mkdir(parents=True, exist_ok=True)
        (tmp_path / "train" / "Cam1" / "depth").mkdir(parents=True, exist_ok=True)

        dataset = RailDualModalDataset(
            train_root=train_root,
            test_root=test_root,
            view_id=1,
            split="val",
            use_patch=True,
            preload=False,
            depth_norm="log"
        )

        assert dataset.use_patch is True
        assert dataset.preload is False
        assert dataset.depth_norm == "log"
        assert dataset.split == "val"

    # ===== 测试用例 TC-DT-005: 采样相关组合 =====

    def test_combination_sample_ratio_with_num(self, tmp_path):
        """
        测试用例ID: TC-DT-005-1
        判定表规则: train_sample_ratio=0.5, train_sample_num=100
        预期结果: train_sample_num 优先级高于 ratio
        """
        train_root = str(tmp_path / "train")
        test_root = str(tmp_path / "test")

        (tmp_path / "train" / "Cam1" / "rgb").mkdir(parents=True, exist_ok=True)
        (tmp_path / "train" / "Cam1" / "depth").mkdir(parents=True, exist_ok=True)

        dataset = RailDualModalDataset(
            train_root=train_root,
            test_root=test_root,
            view_id=1,
            split="train",
            train_sample_ratio=0.5,
            train_sample_num=100
        )

        # train_sample_num 优先级应该更高
        assert dataset.train_sample_num == 100
        assert dataset.train_sample_ratio == 0.5

    def test_combination_sample_ratio_only(self, tmp_path):
        """
        测试用例ID: TC-DT-005-2
        判定表规则: train_sample_ratio=0.8, train_sample_num=None
        预期结果: 使用 ratio 进行采样
        """
        train_root = str(tmp_path / "train")
        test_root = str(tmp_path / "test")

        (tmp_path / "train" / "Cam1" / "rgb").mkdir(parents=True, exist_ok=True)
        (tmp_path / "train" / "Cam1" / "depth").mkdir(parents=True, exist_ok=True)

        dataset = RailDualModalDataset(
            train_root=train_root,
            test_root=test_root,
            view_id=1,
            split="train",
            train_sample_ratio=0.8,
            train_sample_num=None
        )

        assert dataset.train_sample_ratio == 0.8
        assert dataset.train_sample_num is None

    # ===== 测试用例 TC-DT-006: 冲突配置组合 =====

    def test_combination_patch_size_stride_conflict(self, tmp_path):
        """
        测试用例ID: TC-DT-006-1
        判定表规则: use_patch=False, 但设置了 patch_size 和 patch_stride
        预期结果: patch 参数应该被忽略
        """
        train_root = str(tmp_path / "train")
        test_root = str(tmp_path / "test")

        (tmp_path / "train" / "Cam1" / "rgb").mkdir(parents=True, exist_ok=True)
        (tmp_path / "train" / "Cam1" / "depth").mkdir(parents=True, exist_ok=True)

        dataset = RailDualModalDataset(
            train_root=train_root,
            test_root=test_root,
            view_id=1,
            split="train",
            use_patch=False,
            patch_size=512,
            patch_stride=256
        )

        # use_patch=False 时，patch 相关参数应该不生效
        assert dataset.use_patch is False
        assert dataset.patch_size == 512  # 参数被保存但不使用
        assert dataset.patch_stride == 256

    def test_combination_test_with_train_params(self, tmp_path):
        """
        测试用例ID: TC-DT-006-2
        判定表规则: split='test', 但设置了 train_sample_ratio
        预期结果: 训练集参数对测试集不应生效
        """
        train_root = str(tmp_path / "train")
        test_root = str(tmp_path / "test")

        (tmp_path / "test" / "rail_mvtec" / "cam1" / "test" / "good").mkdir(parents=True, exist_ok=True)
        (tmp_path / "test" / "rail_mvtec_depth" / "cam1" / "test" / "good").mkdir(parents=True, exist_ok=True)

        dataset = RailDualModalDataset(
            train_root=train_root,
            test_root=test_root,
            view_id=1,
            split="test",
            train_sample_ratio=0.5  # 对测试集不应生效
        )

        assert dataset.split == "test"
        assert dataset.train_sample_ratio == 0.5  # 参数被保存但测试集不使用
