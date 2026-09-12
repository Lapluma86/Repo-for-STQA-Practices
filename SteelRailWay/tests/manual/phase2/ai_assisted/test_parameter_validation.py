# -*- coding: utf-8 -*-
"""
Module 1 Phase 2: AI-Assisted Testing
Uses Hypothesis for property-based testing to find edge cases

This file supplements manual tests with automated property-based tests
that explore a wider range of input combinations.
"""

import pytest
import sys
from pathlib import Path
from hypothesis import given, strategies as st, settings, assume, HealthCheck

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.rail_dataset import RailDualModalDataset


@pytest.mark.ai_assisted
@pytest.mark.phase2
class TestParameterValidation:
    """Property-based tests for parameter validation"""

    @given(
        view_id=st.integers(min_value=-10, max_value=20),
        split=st.sampled_from(["train", "val", "test", "invalid", ""]),
        img_size=st.integers(min_value=-100, max_value=10000),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_parameter_combinations(self, tmp_path, view_id, split, img_size):
        """Test random parameter combinations"""
        train_root = str(tmp_path / "train")
        test_root = str(tmp_path / "test")

        # Create directories for all possible view_ids
        for vid in range(1, 9):
            (tmp_path / "train" / f"Cam{vid}" / "rgb").mkdir(parents=True, exist_ok=True)
            (tmp_path / "train" / f"Cam{vid}" / "depth").mkdir(parents=True, exist_ok=True)
            (tmp_path / "test" / "rail_mvtec" / f"cam{vid}" / "test" / "good").mkdir(parents=True, exist_ok=True)
            (tmp_path / "test" / "rail_mvtec_depth" / f"cam{vid}" / "test" / "good").mkdir(parents=True, exist_ok=True)

        # Valid combination should succeed
        valid_view = 1 <= view_id <= 8
        valid_split = split in ["train", "val", "test"]
        valid_size = img_size > 0

        if valid_view and valid_split and valid_size:
            dataset = RailDualModalDataset(
                train_root=train_root,
                test_root=test_root,
                view_id=view_id,
                split=split,
                img_size=img_size
            )
            assert dataset.view_id == view_id
            assert dataset.split == split
            assert dataset.img_size == img_size
        else:
            # Invalid combination should raise
            with pytest.raises((ValueError, FileNotFoundError)):
                RailDualModalDataset(
                    train_root=train_root,
                    test_root=test_root,
                    view_id=view_id,
                    split=split,
                    img_size=img_size
                )


@pytest.mark.ai_assisted
@pytest.mark.phase2
class TestEdgeCaseDiscovery:
    """Discover edge cases through random exploration"""

    @given(
        view_id=st.integers(min_value=1, max_value=8),
        train_sample_ratio=st.floats(min_value=-1.0, max_value=2.0, allow_nan=False),
        patch_stride=st.integers(min_value=-50, max_value=2000),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_sampling_edge_cases(self, tmp_path, view_id, train_sample_ratio, patch_stride):
        """Test edge cases for sampling parameters"""
        train_root = str(tmp_path / "train")
        test_root = str(tmp_path / "test")

        (tmp_path / "train" / f"Cam{view_id}" / "rgb").mkdir(parents=True, exist_ok=True)
        (tmp_path / "train" / f"Cam{view_id}" / "depth").mkdir(parents=True, exist_ok=True)

        # Valid range check
        if 0 <= train_sample_ratio <= 1 and patch_stride > 0:
            dataset = RailDualModalDataset(
                train_root=train_root,
                test_root=test_root,
                view_id=view_id,
                split="train",
                train_sample_ratio=train_sample_ratio,
                patch_stride=patch_stride
            )
            assert dataset.train_sample_ratio == train_sample_ratio
            assert dataset.patch_stride == patch_stride
        else:
            with pytest.raises(ValueError):
                RailDualModalDataset(
                    train_root=train_root,
                    test_root=test_root,
                    view_id=view_id,
                    split="train",
                    train_sample_ratio=train_sample_ratio,
                    patch_stride=patch_stride
                )


@pytest.mark.ai_assisted
@pytest.mark.phase2
class TestConfigurationCombinations:
    """Test complex configuration combinations"""

    @given(
        depth_norm=st.sampled_from(["zscore", "minmax", "log", "invalid"]),
        use_patch=st.booleans(),
        preload=st.booleans(),
        sampling_mode=st.sampled_from(["random", "uniform_time", "invalid"]),
    )
    @settings(max_examples=40, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_config_combinations(self, tmp_path, depth_norm, use_patch, preload, sampling_mode):
        """Test various configuration combinations"""
        train_root = str(tmp_path / "train")
        test_root = str(tmp_path / "test")

        (tmp_path / "train" / "Cam1" / "rgb").mkdir(parents=True, exist_ok=True)
        (tmp_path / "train" / "Cam1" / "depth").mkdir(parents=True, exist_ok=True)

        # Valid combinations
        if depth_norm in ["zscore", "minmax", "log"] and sampling_mode in ["random", "uniform_time"]:
            dataset = RailDualModalDataset(
                train_root=train_root,
                test_root=test_root,
                view_id=1,
                split="train",
                depth_norm=depth_norm,
                use_patch=use_patch,
                preload=preload,
                sampling_mode=sampling_mode
            )
            assert dataset.depth_norm == depth_norm
            assert dataset.use_patch == use_patch
            assert dataset.preload == preload
            assert dataset.sampling_mode == sampling_mode
        else:
            with pytest.raises(ValueError):
                RailDualModalDataset(
                    train_root=train_root,
                    test_root=test_root,
                    view_id=1,
                    split="train",
                    depth_norm=depth_norm,
                    use_patch=use_patch,
                    preload=preload,
                    sampling_mode=sampling_mode
                )


@pytest.mark.ai_assisted
@pytest.mark.phase2
class TestRegressionDiscovery:
    """Regression tests to ensure fixes remain in place"""

    def test_patch_stride_zero_still_rejected(self, tmp_path):
        """Regression: Ensure DEF-004 fix is not reverted"""
        train_root = str(tmp_path / "train")
        test_root = str(tmp_path / "test")

        (tmp_path / "train" / "Cam1" / "rgb").mkdir(parents=True, exist_ok=True)
        (tmp_path / "train" / "Cam1" / "depth").mkdir(parents=True, exist_ok=True)

        with pytest.raises(ValueError, match="patch_stride must be a positive integer"):
            RailDualModalDataset(
                train_root=train_root,
                test_root=test_root,
                view_id=1,
                split="train",
                patch_stride=0
            )

    def test_invalid_view_id_still_rejected(self, tmp_path):
        """Regression: Ensure DEF-001 fix is not reverted"""
        train_root = str(tmp_path / "train")
        test_root = str(tmp_path / "test")

        with pytest.raises(ValueError, match="view_id must be an integer between 1 and 8"):
            RailDualModalDataset(
                train_root=train_root,
                test_root=test_root,
                view_id=0,
                split="train"
            )

    def test_invalid_depth_norm_still_rejected(self, tmp_path):
        """Regression: Ensure DEF-003 fix is not reverted"""
        train_root = str(tmp_path / "train")
        test_root = str(tmp_path / "test")

        (tmp_path / "train" / "Cam1" / "rgb").mkdir(parents=True, exist_ok=True)
        (tmp_path / "train" / "Cam1" / "depth").mkdir(parents=True, exist_ok=True)

        with pytest.raises(ValueError, match="depth_norm must be one of"):
            RailDualModalDataset(
                train_root=train_root,
                test_root=test_root,
                view_id=1,
                split="train",
                depth_norm="invalid_method"
            )

    def test_empty_list_in_loss_functions(self, tmp_path):
        """Regression: Ensure DEF-WB-001 fix is not reverted"""
        from utils.losses import loss_l2, loss_distil, loss_distil_p, loss_distil_pixel

        # All loss functions should handle empty lists gracefully
        for loss_fn in [loss_l2, loss_distil, loss_distil_p, loss_distil_pixel]:
            result = loss_fn([], [])
            assert isinstance(result, type(result))
            assert result.item() == 0.0
