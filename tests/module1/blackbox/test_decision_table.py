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

from datasets.rail_dataset import RailDualModalDataset


@pytest.mark.blackbox
@pytest.mark.module1
class TestDecisionTable:
    """判定表测试类"""


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
