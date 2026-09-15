"""AI辅助属性/变形测试：数值、缓存和访问顺序，预期来自明确数学性质。
函数数量不等同Hypothesis生成示例次数。
"""
import cv2
import numpy as np
import pytest
import torch
from hypothesis import given, settings, strategies as st, HealthCheck

pytestmark = [pytest.mark.module2, pytest.mark.ai_assisted]
property_test = settings(max_examples=12, deadline=None,
                         suppress_health_check=[HealthCheck.function_scoped_fixture])


def replace_depth(dataset, values):
    assert cv2.imwrite(dataset.samples[0]["depth_path"], values.astype(np.uint16))


@property_test
@given(value=st.integers(0, 1000))
def test_minmax_constant_is_zero(rail_factory, value):
    dataset = rail_factory(depth_norm="minmax")
    replace_depth(dataset, np.full((16, 24), value))
    assert torch.count_nonzero(dataset[0]["depth"]) == 0


@property_test
@given(value=st.integers(1, 1000))
def test_zscore_constant_is_finite_zero(rail_factory, value):
    dataset = rail_factory(depth_norm="zscore")
    replace_depth(dataset, np.full((16, 24), value))
    assert torch.count_nonzero(dataset[0]["depth"]) == 0


def test_all_zero_depth_is_finite(rail_factory):
    for norm in ("zscore", "minmax", "log"):
        dataset = rail_factory(depth_norm=norm)
        replace_depth(dataset, np.zeros((16, 24)))
        assert torch.count_nonzero(dataset[0]["depth"]) == 0
        assert torch.isfinite(dataset[0]["depth"]).all()


@property_test
@given(value=st.integers(0, 65535))
def test_log_matches_known_value(rail_factory, value):
    dataset = rail_factory(depth_norm="log")
    replace_depth(dataset, np.full((16, 24), value))
    torch.testing.assert_close(dataset[0]["depth"], torch.full((3, 8, 8), float(np.log1p(value))))


def test_minmax_range_and_order(rail_factory):
    dataset = rail_factory(depth_norm="minmax")
    result = dataset[0]["depth"][0]
    assert result.min() >= 0 and result.max() <= 1
    assert torch.all(result[1:] > result[:-1])


def test_zscore_patch_independent_of_access_order(rail_factory):
    forward = rail_factory(use_patch=True)
    reverse = rail_factory(use_patch=True)
    a = {i: forward[i]["depth"] for i in (0, 1)}
    b = {i: reverse[i]["depth"] for i in (1, 0)}
    for i in (0, 1):
        torch.testing.assert_close(a[i], b[i])


def test_zscore_patch_matches_full_frame_statistics(rail_factory):
    dataset = rail_factory(use_patch=True)
    full = np.arange(1, 385, dtype=np.float32).reshape(16, 24)
    for i in (0, 1):
        expected = (full[i*8:i*8+8, 8:16] - full.mean()) / (full.std() + 1e-6)
        torch.testing.assert_close(dataset[i]["depth"][0], torch.from_numpy(expected))


def test_patch_preload_matches_lazy(rail_factory):
    lazy = rail_factory(use_patch=True)
    cached = rail_factory(use_patch=True, preload=True)
    for i in (1, 0):
        torch.testing.assert_close(lazy[i]["depth"], cached[i]["depth"])


def test_depth_channels_are_identical(rail_factory):
    for norm in ("zscore", "minmax", "log"):
        dataset = rail_factory(depth_norm=norm)
        depth = dataset[0]["depth"]
        torch.testing.assert_close(depth[0], depth[1])
        torch.testing.assert_close(depth[1], depth[2])
