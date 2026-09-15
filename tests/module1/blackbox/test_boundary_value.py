# -*- coding: utf-8 -*-
"""
Blackbox Test - Boundary Value Analysis
Tests critical parameter boundary values
"""

import os
from pathlib import Path
import pytest

from datasets.rail_dataset import RailDualModalDataset


@pytest.mark.blackbox
@pytest.mark.module1
class TestBoundaryValueAnalysis:
    """Boundary value analysis test class"""


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
