# -*- coding: utf-8 -*-
"""
Blackbox Test - Equivalence Partitioning
Tests RailDualModalDataset input parameter equivalence classes
"""

import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.rail_dataset import RailDualModalDataset


@pytest.mark.blackbox
@pytest.mark.phase1
class TestEquivalencePartitioning:
    """Equivalence partitioning test class"""

    # ===== TC-EP-001: view_id equivalence class =====

    def test_view_id_valid_range(self, tmp_path):
        """TC-EP-001-1: Valid range 1-8 should create dataset successfully"""
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

    def test_view_id_boundary_lower(self, tmp_path):
        """TC-EP-001-2: Lower boundary value 1"""
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

        assert dataset is not None

    def test_view_id_boundary_upper(self, tmp_path):
        """TC-EP-001-3: Upper boundary value 8"""
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

    def test_view_id_invalid_zero(self, tmp_path):
        """TC-EP-001-4: view_id=0 should raise ValueError"""
        train_root = str(tmp_path / "train")
        test_root = str(tmp_path / "test")

        with pytest.raises(ValueError, match="view_id must be an integer between 1 and 8"):
            RailDualModalDataset(
                train_root=train_root,
                test_root=test_root,
                view_id=0,
                split="train"
            )

    def test_view_id_invalid_negative(self, tmp_path):
        """TC-EP-001-5: Negative view_id should raise ValueError"""
        train_root = str(tmp_path / "train")
        test_root = str(tmp_path / "test")

        with pytest.raises(ValueError, match="view_id must be an integer between 1 and 8"):
            RailDualModalDataset(
                train_root=train_root,
                test_root=test_root,
                view_id=-1,
                split="train"
            )

    def test_view_id_invalid_out_of_range(self, tmp_path):
        """TC-EP-001-6: view_id > 8 should raise ValueError"""
        train_root = str(tmp_path / "train")
        test_root = str(tmp_path / "test")

        with pytest.raises(ValueError, match="view_id must be an integer between 1 and 8"):
            RailDualModalDataset(
                train_root=train_root,
                test_root=test_root,
                view_id=9,
                split="train"
            )

    # ===== TC-EP-002: split equivalence class =====

    def test_split_valid_train(self, tmp_path):
        """TC-EP-002-1: split='train' valid input"""
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

        assert dataset.split == "train"

    def test_split_valid_val(self, tmp_path):
        """TC-EP-002-2: split='val' valid input"""
        train_root = str(tmp_path / "train")
        test_root = str(tmp_path / "test")

        (tmp_path / "train" / "Cam1" / "rgb").mkdir(parents=True, exist_ok=True)
        (tmp_path / "train" / "Cam1" / "depth").mkdir(parents=True, exist_ok=True)

        dataset = RailDualModalDataset(
            train_root=train_root,
            test_root=test_root,
            view_id=1,
            split="val"
        )

        assert dataset.split == "val"

    def test_split_valid_test(self, tmp_path):
        """TC-EP-002-3: split='test' valid input"""
        train_root = str(tmp_path / "train")
        test_root = str(tmp_path / "test")

        (tmp_path / "test" / "rail_mvtec" / "cam1" / "test" / "good").mkdir(parents=True, exist_ok=True)
        (tmp_path / "test" / "rail_mvtec_depth" / "cam1" / "test" / "good").mkdir(parents=True, exist_ok=True)

        dataset = RailDualModalDataset(
            train_root=train_root,
            test_root=test_root,
            view_id=1,
            split="test"
        )

        assert dataset.split == "test"

    def test_split_invalid_empty(self, tmp_path):
        """TC-EP-002-4: Empty string should raise ValueError"""
        train_root = str(tmp_path / "train")
        test_root = str(tmp_path / "test")

        with pytest.raises(ValueError, match="split must be one of"):
            RailDualModalDataset(
                train_root=train_root,
                test_root=test_root,
                view_id=1,
                split=""
            )

    def test_split_invalid_unknown(self, tmp_path):
        """TC-EP-002-5: Invalid split value should raise ValueError"""
        train_root = str(tmp_path / "train")
        test_root = str(tmp_path / "test")

        with pytest.raises(ValueError, match="split must be one of"):
            RailDualModalDataset(
                train_root=train_root,
                test_root=test_root,
                view_id=1,
                split="unknown"
            )

    # ===== TC-EP-003: img_size equivalence class =====

    def test_img_size_valid_default(self, tmp_path):
        """TC-EP-003-1: Default value 256"""
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

        assert dataset.img_size == 256

    def test_img_size_valid_custom(self, tmp_path):
        """TC-EP-003-2: Custom value 512"""
        train_root = str(tmp_path / "train")
        test_root = str(tmp_path / "test")

        (tmp_path / "train" / "Cam1" / "rgb").mkdir(parents=True, exist_ok=True)
        (tmp_path / "train" / "Cam1" / "depth").mkdir(parents=True, exist_ok=True)

        dataset = RailDualModalDataset(
            train_root=train_root,
            test_root=test_root,
            view_id=1,
            split="train",
            img_size=512
        )

        assert dataset.img_size == 512

    def test_img_size_invalid_zero(self, tmp_path):
        """TC-EP-003-3: Zero should raise ValueError"""
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

    def test_img_size_invalid_negative(self, tmp_path):
        """TC-EP-003-4: Negative value should raise ValueError"""
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

    # ===== TC-EP-004: depth_norm equivalence class =====

    def test_depth_norm_valid_zscore(self, tmp_path):
        """TC-EP-004-1: Valid 'zscore'"""
        train_root = str(tmp_path / "train")
        test_root = str(tmp_path / "test")

        (tmp_path / "train" / "Cam1" / "rgb").mkdir(parents=True, exist_ok=True)
        (tmp_path / "train" / "Cam1" / "depth").mkdir(parents=True, exist_ok=True)

        dataset = RailDualModalDataset(
            train_root=train_root,
            test_root=test_root,
            view_id=1,
            split="train",
            depth_norm="zscore"
        )

        assert dataset.depth_norm == "zscore"

    def test_depth_norm_valid_minmax(self, tmp_path):
        """TC-EP-004-2: Valid 'minmax'"""
        train_root = str(tmp_path / "train")
        test_root = str(tmp_path / "test")

        (tmp_path / "train" / "Cam1" / "rgb").mkdir(parents=True, exist_ok=True)
        (tmp_path / "train" / "Cam1" / "depth").mkdir(parents=True, exist_ok=True)

        dataset = RailDualModalDataset(
            train_root=train_root,
            test_root=test_root,
            view_id=1,
            split="train",
            depth_norm="minmax"
        )

        assert dataset.depth_norm == "minmax"

    def test_depth_norm_valid_log(self, tmp_path):
        """TC-EP-004-3: Valid 'log'"""
        train_root = str(tmp_path / "train")
        test_root = str(tmp_path / "test")

        (tmp_path / "train" / "Cam1" / "rgb").mkdir(parents=True, exist_ok=True)
        (tmp_path / "train" / "Cam1" / "depth").mkdir(parents=True, exist_ok=True)

        dataset = RailDualModalDataset(
            train_root=train_root,
            test_root=test_root,
            view_id=1,
            split="train",
            depth_norm="log"
        )

        assert dataset.depth_norm == "log"

    def test_depth_norm_invalid(self, tmp_path):
        """TC-EP-004-4: Invalid value should raise ValueError"""
        train_root = str(tmp_path / "train")
        test_root = str(tmp_path / "test")

        with pytest.raises(ValueError, match="depth_norm must be one of"):
            RailDualModalDataset(
                train_root=train_root,
                test_root=test_root,
                view_id=1,
                split="train",
                depth_norm="invalid"
            )

    # ===== TC-EP-005: train_sample_ratio equivalence class =====

    def test_train_sample_ratio_valid_full(self, tmp_path):
        """TC-EP-005-1: Value 1.0 uses all data"""
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

    def test_train_sample_ratio_valid_half(self, tmp_path):
        """TC-EP-005-2: Value 0.5 uses half data"""
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

    def test_train_sample_ratio_boundary_zero(self, tmp_path):
        """TC-EP-005-3: Boundary value 0.0"""
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

    def test_train_sample_ratio_invalid_negative(self, tmp_path):
        """TC-EP-005-4: Negative value should raise ValueError"""
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

    def test_train_sample_ratio_invalid_greater_than_one(self, tmp_path):
        """TC-EP-005-5: Value > 1 should raise ValueError"""
        train_root = str(tmp_path / "train")
        test_root = str(tmp_path / "test")

        with pytest.raises(ValueError, match="train_sample_ratio must be in"):
            RailDualModalDataset(
                train_root=train_root,
                test_root=test_root,
                view_id=1,
                split="train",
                train_sample_ratio=1.5
            )

    # ===== TC-EP-006: patch_stride equivalence class =====

    def test_patch_stride_valid(self, tmp_path):
        """TC-EP-006-1: Valid positive value"""
        train_root = str(tmp_path / "train")
        test_root = str(tmp_path / "test")

        (tmp_path / "train" / "Cam1" / "rgb").mkdir(parents=True, exist_ok=True)
        (tmp_path / "train" / "Cam1" / "depth").mkdir(parents=True, exist_ok=True)

        dataset = RailDualModalDataset(
            train_root=train_root,
            test_root=test_root,
            view_id=1,
            split="train",
            patch_stride=850
        )

        assert dataset.patch_stride == 850

    def test_patch_stride_invalid_zero(self, tmp_path):
        """TC-EP-006-2: Zero should raise ValueError (DEF-004 fix)"""
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

    def test_patch_stride_invalid_negative(self, tmp_path):
        """TC-EP-006-3: Negative value should raise ValueError"""
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
