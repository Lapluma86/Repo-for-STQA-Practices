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


def test_scan_then_read(rail_factory):
    dataset = rail_factory(count=2)
    assert len(dataset) == 2  # 扫描发生在构造时，__len__ 不负责扫描。
    assert_item(dataset[0])
    assert dataset[1]["frame_id"] == "frame_001"


def test_lazy_preload_equivalence(rail_factory):
    lazy = rail_factory()
    preload = rail_factory(preload=True)
    assert len(lazy) == len(preload) == 1
    assert len(preload.rgb_cache) == len(preload.depth_cache) == 1
    for key in ("intensity", "depth", "gt"):
        torch.testing.assert_close(lazy[0][key], preload[0][key])


def test_state_transition_error_recovery(rail_factory):
    dataset = rail_factory()
    assert len(dataset) == 1
    with pytest.raises(IndexError):
        dataset[999]
    assert_item(dataset[0])
