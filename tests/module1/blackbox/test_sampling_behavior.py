"""采样行为：数量优先级、边界及帧级隔离（AI辅助补充）。"""
import pytest
from datasets.rail_dataset import RailDualModalDataset

pytestmark = [pytest.mark.module1, pytest.mark.blackbox]


def frames(dataset):
    return [s["frame_id"] for s in dataset.samples]


def test_count_overrides_ratio(rail_factory):
    dataset = rail_factory(count=10, train_sample_num=4, train_sample_ratio=0.9)
    assert frames(dataset) == ["frame_000", "frame_003", "frame_006", "frame_009"]


def test_random_sampling_reproducible_unique(rail_factory):
    a = rail_factory(count=20, train_sample_num=6, sampling_mode="random", random_seed=7)
    b = rail_factory(count=20, train_sample_num=6, sampling_mode="random", random_seed=7)
    assert frames(a) == frames(b)
    assert len(set(frames(a))) == 6
