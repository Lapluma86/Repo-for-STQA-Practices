# -*- coding: utf-8 -*-
"""
Blackbox Test - Boundary Value Analysis
Tests critical parameter boundary values
"""

import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.rail_dataset import RailDualModalDataset


@pytest.mark.blackbox
@pytest.mark.phase1
class TestBoundaryValueAnalysis:
    """Boundary value analysis test class"""

    # ===== TC-BV-001: img_size boundary tests =====

    def test_img_size_minimum_value(self, tmp_path):
        """TC-BV-001-1: Minimum reasonable value 1"""
        train_root = str(tmp_path / "train")
        test_root = str(tmp_path / "test")

        (tmp_path / "train" / "Cam1" / "rgb").mkdir(parents=True, exist_ok=True)
        (tmp_path / "train" / "Cam1" / "depth").mkdir(parents=True, exist_ok=True)

        dataset = RailDualModalDataset(
            train_root=train_root,
            test_root=test_root,
            view_id=1,
            split="train",
            img_size=1
        )

        assert dataset.img_size == 1

    def test_img_size_minimum_plus_one(self, tmp_path):
        """TC-BV-001-2: Minimum value + 1 = 2"""
        train_root = str(tmp_path / "train")
        test_root = str(tmp_path / "test")

        (tmp_path / "train" / "Cam1" / "rgb").mkdir(parents=True, exist_ok=True)
        (tmp_path / "train" / "Cam1" / "depth").mkdir(parents=True, exist_ok=True)

        dataset = RailDualModalDataset(
            train_root=train_root,
            test_root=test_root,
            view_id=1,
            split="train",
            img_size=2
        )

        assert dataset.img_size == 2

    def test_img_size_common_values(self, tmp_path):
        """TC-BV-001-3: Common values 224, 256, 512, 1024"""
        train_root = str(tmp_path / "train")
        test_root = str(tmp_path / "test")

        (tmp_path / "train" / "Cam1" / "rgb").mkdir(parents=True, exist_ok=True)
        (tmp_path / "train" / "Cam1" / "depth").mkdir(parents=True, exist_ok=True)

        for size in [224, 256, 512, 1024]:
            dataset = RailDualModalDataset(
                train_root=train_root,
                test_root=test_root,
                view_id=1,
                split="train",
                img_size=size
            )
            assert dataset.img_size == size

    def test_img_size_very_large(self, tmp_path):
        """TC-BV-001-4: Very large value 10000"""
        train_root = str(tmp_path / "train")
        test_root = str(tmp_path / "test")

        (tmp_path / "train" / "Cam1" / "rgb").mkdir(parents=True, exist_ok=True)
        (tmp_path / "train" / "Cam1" / "depth").mkdir(parents=True, exist_ok=True)

        dataset = RailDualModalDataset(
            train_root=train_root,
            test_root=test_root,
            view_id=1,
            split="train",
            img_size=10000
        )

        assert dataset.img_size == 10000

    def test_img_size_zero_boundary(self, tmp_path):
        """TC-BV-001-5: Zero should raise ValueError"""
        train_root = str(tmp_path / "train")
        test_root = str(tmp_path / "test")

        with pytest.raises(ValueError, match="img_size must be a positive integer"):
            RailDualModalDataset(
                train_root=train_root,
                test_root=test_root,
                view_id=1,
                split="train",
                img_size=0
            )

    def test_img_size_negative_boundary(self, tmp_path):
        """TC-BV-001-6: Negative value should raise ValueError"""
        train_root = str(tmp_path / "train")
        test_root = str(tmp_path / "test")

        with pytest.raises(ValueError, match="img_size must be a positive integer"):
            RailDualModalDataset(
                train_root=train_root,
                test_root=test_root,
                view_id=1,
                split="train",
                img_size=-1
            )

    # ===== TC-BV-002: patch_size boundary tests =====

    def test_patch_size_minimum(self, tmp_path):
        """TC-BV-002-1: Minimum value 1"""
        train_root = str(tmp_path / "train")
        test_root = str(tmp_path / "test")

        (tmp_path / "train" / "Cam1" / "rgb").mkdir(parents=True, exist_ok=True)
        (tmp_path / "train" / "Cam1" / "depth").mkdir(parents=True, exist_ok=True)

        dataset = RailDualModalDataset(
            train_root=train_root,
            test_root=test_root,
            view_id=1,
            split="train",
            patch_size=1
        )

        assert dataset.patch_size == 1

    def test_patch_size_default(self, tmp_path):
        """TC-BV-002-2: Default value 900"""
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

        assert dataset.patch_size == 900

    def test_patch_size_zero_invalid(self, tmp_path):
        """TC-BV-002-3: Zero should raise ValueError"""
        train_root = str(tmp_path / "train")
        test_root = str(tmp_path / "test")

        with pytest.raises(ValueError, match="patch_size must be a positive integer"):
            RailDualModalDataset(
                train_root=train_root,
                test_root=test_root,
                view_id=1,
                split="train",
                patch_size=0
            )

    # ===== TC-BV-003: patch_stride boundary tests =====

    def test_patch_stride_valid_values(self, tmp_path):
        """TC-BV-003-1: Valid positive values"""
        train_root = str(tmp_path / "train")
        test_root = str(tmp_path / "test")

        (tmp_path / "train" / "Cam1" / "rgb").mkdir(parents=True, exist_ok=True)
        (tmp_path / "train" / "Cam1" / "depth").mkdir(parents=True, exist_ok=True)

        for stride in [1, 100, 850, 900]:
            dataset = RailDualModalDataset(
                train_root=train_root,
                test_root=test_root,
                view_id=1,
                split="train",
                patch_stride=stride
            )
            assert dataset.patch_stride == stride

    def test_patch_stride_zero_invalid(self, tmp_path):
        """TC-BV-003-2: Zero should raise ValueError (DEF-004 fix)"""
        train_root = str(tmp_path / "train")
        test_root = str(tmp_path / "test")

        with pytest.raises(ValueError, match="patch_stride must be a positive integer"):
            RailDualModalDataset(
                train_root=train_root,
                test_root=test_root,
                view_id=1,
                split="train",
                patch_stride=0
            )

    def test_patch_stride_negative_invalid(self, tmp_path):
        """TC-BV-003-3: Negative should raise ValueError"""
        train_root = str(tmp_path / "train")
        test_root = str(tmp_path / "test")

        with pytest.raises(ValueError, match="patch_stride must be a positive integer"):
            RailDualModalDataset(
                train_root=train_root,
                test_root=test_root,
                view_id=1,
                split="train",
                patch_stride=-100
            )

    def test_patch_stride_equals_patch_size(self, tmp_path):
        """TC-BV-003-4: Equal to patch_size (no overlap)"""
        train_root = str(tmp_path / "train")
        test_root = str(tmp_path / "test")

        (tmp_path / "train" / "Cam1" / "rgb").mkdir(parents=True, exist_ok=True)
        (tmp_path / "train" / "Cam1" / "depth").mkdir(parents=True, exist_ok=True)

        dataset = RailDualModalDataset(
            train_root=train_root,
            test_root=test_root,
            view_id=1,
            split="train",
            patch_size=900,
            patch_stride=900
        )

        assert dataset.patch_stride == dataset.patch_size

    # ===== TC-BV-004: train_sample_ratio boundary tests =====

    def test_train_sample_ratio_zero(self, tmp_path):
        """TC-BV-004-1: Zero boundary"""
        train_root = str(tmp_path / "train")
        test_root = str(tmp_path / "test")

        (tmp_path / "train" / "Cam1" / "rgb").mkdir(parents=True, exist_ok=True)
        (tmp_path / "train" / "Cam1" / "depth").mkdir(parents=True, exist_ok=True)

        dataset = RailDualModalDataset(
            train_root=train_root,
            test_root=test_root,
            view_id=1,
            split="train",
            train_sample_ratio=0.0
        )

        assert dataset.train_sample_ratio == 0.0

    def test_train_sample_ratio_near_zero(self, tmp_path):
        """TC-BV-004-2: Near zero 0.001"""
        train_root = str(tmp_path / "train")
        test_root = str(tmp_path / "test")

        (tmp_path / "train" / "Cam1" / "rgb").mkdir(parents=True, exist_ok=True)
        (tmp_path / "train" / "Cam1" / "depth").mkdir(parents=True, exist_ok=True)

        dataset = RailDualModalDataset(
            train_root=train_root,
            test_root=test_root,
            view_id=1,
            split="train",
            train_sample_ratio=0.001
        )

        assert dataset.train_sample_ratio == 0.001

    def test_train_sample_ratio_half(self, tmp_path):
        """TC-BV-004-3: Middle value 0.5"""
        train_root = str(tmp_path / "train")
        test_root = str(tmp_path / "test")

        (tmp_path / "train" / "Cam1" / "rgb").mkdir(parents=True, exist_ok=True)
        (tmp_path / "train" / "Cam1" / "depth").mkdir(parents=True, exist_ok=True)

        dataset = RailDualModalDataset(
            train_root=train_root,
            test_root=test_root,
            view_id=1,
            split="train",
            train_sample_ratio=0.5
        )

        assert dataset.train_sample_ratio == 0.5

    def test_train_sample_ratio_near_one(self, tmp_path):
        """TC-BV-004-4: Near one 0.999"""
        train_root = str(tmp_path / "train")
        test_root = str(tmp_path / "test")

        (tmp_path / "train" / "Cam1" / "rgb").mkdir(parents=True, exist_ok=True)
        (tmp_path / "train" / "Cam1" / "depth").mkdir(parents=True, exist_ok=True)

        dataset = RailDualModalDataset(
            train_root=train_root,
            test_root=test_root,
            view_id=1,
            split="train",
            train_sample_ratio=0.999
        )

        assert dataset.train_sample_ratio == 0.999

    def test_train_sample_ratio_one(self, tmp_path):
        """TC-BV-004-5: One boundary"""
        train_root = str(tmp_path / "train")
        test_root = str(tmp_path / "test")

        (tmp_path / "train" / "Cam1" / "rgb").mkdir(parents=True, exist_ok=True)
        (tmp_path / "train" / "Cam1" / "depth").mkdir(parents=True, exist_ok=True)

        dataset = RailDualModalDataset(
            train_root=train_root,
            test_root=test_root,
            view_id=1,
            split="train",
            train_sample_ratio=1.0
        )

        assert dataset.train_sample_ratio == 1.0

    def test_train_sample_ratio_above_one(self, tmp_path):
        """TC-BV-004-6: Above one should raise ValueError"""
        train_root = str(tmp_path / "train")
        test_root = str(tmp_path / "test")

        with pytest.raises(ValueError, match="train_sample_ratio must be in"):
            RailDualModalDataset(
                train_root=train_root,
                test_root=test_root,
                view_id=1,
                split="train",
                train_sample_ratio=1.1
            )

    def test_train_sample_ratio_negative(self, tmp_path):
        """TC-BV-004-7: Negative should raise ValueError"""
        train_root = str(tmp_path / "train")
        test_root = str(tmp_path / "test")

        with pytest.raises(ValueError, match="train_sample_ratio must be in"):
            RailDualModalDataset(
                train_root=train_root,
                test_root=test_root,
                view_id=1,
                split="train",
                train_sample_ratio=-0.5
            )

    # ===== TC-BV-005: view_id boundary tests =====

    def test_view_id_minimum(self, tmp_path):
        """TC-BV-005-1: Minimum valid value 1"""
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

        assert dataset.view_id == 1

    def test_view_id_middle(self, tmp_path):
        """TC-BV-005-2: Middle value 4"""
        train_root = str(tmp_path / "train")
        test_root = str(tmp_path / "test")

        (tmp_path / "train" / "Cam4" / "rgb").mkdir(parents=True, exist_ok=True)
        (tmp_path / "train" / "Cam4" / "depth").mkdir(parents=True, exist_ok=True)

        dataset = RailDualModalDataset(
            train_root=train_root,
            test_root=test_root,
            view_id=4,
            split="train"
        )

        assert dataset.view_id == 4

    def test_view_id_maximum(self, tmp_path):
        """TC-BV-005-3: Maximum value 8"""
        train_root = str(tmp_path / "train")
        test_root = str(tmp_path / "test")

        (tmp_path / "train" / "Cam8" / "rgb").mkdir(parents=True, exist_ok=True)
        (tmp_path / "train" / "Cam8" / "depth").mkdir(parents=True, exist_ok=True)

        dataset = RailDualModalDataset(
            train_root=train_root,
            test_root=test_root,
            view_id=8,
            split="train"
        )

        assert dataset.view_id == 8

    def test_view_id_zero_invalid(self, tmp_path):
        """TC-BV-005-4: Zero should raise ValueError"""
        train_root = str(tmp_path / "train")
        test_root = str(tmp_path / "test")

        with pytest.raises(ValueError, match="view_id must be an integer between 1 and 8"):
            RailDualModalDataset(
                train_root=train_root,
                test_root=test_root,
                view_id=0,
                split="train"
            )

    def test_view_id_above_maximum_invalid(self, tmp_path):
        """TC-BV-005-5: Above maximum should raise ValueError"""
        train_root = str(tmp_path / "train")
        test_root = str(tmp_path / "test")

        with pytest.raises(ValueError, match="view_id must be an integer between 1 and 8"):
            RailDualModalDataset(
                train_root=train_root,
                test_root=test_root,
                view_id=9,
                split="train"
            )

    # ===== TC-BV-006: path parameter boundary tests =====

    def test_nonexistent_train_root(self, tmp_path):
        """TC-BV-006-1: Nonexistent path should raise FileNotFoundError"""
        train_root = str(tmp_path / "nonexistent_dir")
        test_root = str(tmp_path / "test")

        with pytest.raises(FileNotFoundError, match="Training directory not found"):
            RailDualModalDataset(
                train_root=train_root,
                test_root=test_root,
                view_id=1,
                split="train"
            )

    def test_very_long_path(self, tmp_path):
        """TC-BV-006-2: Very long path name"""
        long_name = "a" * 200
        train_root = str(tmp_path / long_name)
        test_root = str(tmp_path / "test")

        (tmp_path / long_name / "Cam1" / "rgb").mkdir(parents=True, exist_ok=True)
        (tmp_path / long_name / "Cam1" / "depth").mkdir(parents=True, exist_ok=True)

        dataset = RailDualModalDataset(
            train_root=train_root,
            test_root=test_root,
            view_id=1,
            split="train"
        )

        assert len(dataset.train_root) > 200
