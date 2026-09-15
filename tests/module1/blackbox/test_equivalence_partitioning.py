# -*- coding: utf-8 -*-
"""
Blackbox Test - Equivalence Partitioning
Tests RailDualModalDataset input parameter equivalence classes
"""

import pytest

from datasets.rail_dataset import RailDualModalDataset


@pytest.mark.blackbox
@pytest.mark.module1
class TestEquivalencePartitioning:
    """Equivalence partitioning test class"""


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
