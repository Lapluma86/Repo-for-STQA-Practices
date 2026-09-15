# -*- coding: utf-8 -*-
"""数据访问生命周期与配置一致性；配置对比不视为对象状态切换。
2026-09-15 AI 辅助修订：保留12项设计目的，实际读取数据并验证结果。
"""
import pytest
import torch

pytestmark = [pytest.mark.blackbox, pytest.mark.module1]


def assert_item(item):
    assert set(item) == {"intensity", "depth", "label", "view_id", "frame_id", "patch_idx", "gt"}
    for key in ("intensity", "depth"):
        assert item[key].shape == (3, 8, 8)
        assert item[key].dtype == torch.float32
        assert torch.isfinite(item[key]).all()
    assert item["gt"].shape == (8, 8)
    assert item["label"] == 0
    assert item["view_id"] == 1
    assert torch.count_nonzero(item["gt"]) == 0


def test_state_uninitialized_to_initialized(rail_factory):
    dataset = rail_factory()
    assert len(dataset) == 1
    assert dataset.samples[0]["frame_id"] == "frame_000"


def test_configuration_independence(rail_factory):
    train = rail_factory(count=10, train_val_test_split=[0.8, 0.2, 0])
    val = rail_factory(count=10, split="val", train_val_test_split=[0.8, 0.2, 0])
    assert len(train) == 8
    assert len(val) == 2
    assert {s["frame_id"] for s in train.samples}.isdisjoint(s["frame_id"] for s in val.samples)


def test_scan_then_read(rail_factory):
    dataset = rail_factory(count=2)
    assert len(dataset) == 2  # 扫描发生在构造时，__len__ 不负责扫描。
    assert_item(dataset[0])
    assert dataset[1]["frame_id"] == "frame_001"


def test_state_loading_to_ready(rail_factory):
    dataset = rail_factory()
    assert len(dataset) == 1
    assert_item(dataset[0])


def test_state_loading_to_error(rail_factory):
    dataset = rail_factory()
    with pytest.raises(IndexError):
        dataset[len(dataset)]


def test_train_val_split_reproducible(rail_factory):
    first = rail_factory(count=10, train_val_test_split=[0.8, 0.2, 0])
    second = rail_factory(count=10, train_val_test_split=[0.8, 0.2, 0])
    assert len(first) == 8
    assert first.samples == second.samples


def test_test_mode_reads_good_and_broken(rail_factory, tmp_path):
    import cv2
    import numpy as np
    rail_factory()
    for category in ("good", "broken"):
        rgb = tmp_path / "test/rail_mvtec/cam1/test" / category
        depth = tmp_path / "test/rail_mvtec_depth/cam1/test" / category
        rgb.mkdir(parents=True)
        depth.mkdir(parents=True)
        assert cv2.imwrite(str(rgb / "frame.jpg"), np.full((16, 24, 3), 50, np.uint8))
        assert cv2.imwrite(str(depth / "frame.tiff"), np.ones((16, 24), np.uint16))
    gt_dir = tmp_path / "test/rail_mvtec/cam1/ground_truth/broken"
    gt_dir.mkdir(parents=True)
    assert cv2.imwrite(str(gt_dir / "frame.png"), np.full((16, 24), 255, np.uint8))
    dataset = rail_factory(split="test")
    assert len(dataset) == 2
    items = {dataset[i]["label"]: dataset[i] for i in range(2)}
    assert set(items) == {0, 1}
    assert torch.count_nonzero(items[0]["gt"]) == 0
    assert torch.all(items[1]["gt"] == 1)


def test_lazy_preload_equivalence(rail_factory):
    lazy = rail_factory()
    preload = rail_factory(preload=True)
    assert len(lazy) == len(preload) == 1
    assert len(preload.rgb_cache) == len(preload.depth_cache) == 1
    for key in ("intensity", "depth", "gt"):
        torch.testing.assert_close(lazy[0][key], preload[0][key])


def test_patch_configuration_changes_actual_items(rail_factory):
    full = rail_factory()
    patches = rail_factory(use_patch=True)
    assert len(full) == 1
    assert len(patches) == 2
    assert [patches[i]["patch_idx"] for i in range(2)] == [0, 1]
    assert_item(patches[0])


def test_state_transition_sequence(rail_factory):
    dataset = rail_factory()
    assert len(dataset) == 1
    assert_item(dataset[0])
    with pytest.raises(IndexError):
        dataset[1]
    assert_item(dataset[0])


def test_state_transition_repeated_access(rail_factory):
    dataset = rail_factory()
    assert len(dataset) == 1
    first = dataset[0]
    for _ in range(2):
        item = dataset[0]
        for key in ("intensity", "depth", "gt"):
            torch.testing.assert_close(first[key], item[key])


def test_state_transition_error_recovery(rail_factory):
    dataset = rail_factory()
    assert len(dataset) == 1
    with pytest.raises(IndexError):
        dataset[999]
    assert_item(dataset[0])
