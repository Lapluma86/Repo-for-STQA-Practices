"""采样行为：数量优先级、边界及帧级隔离（AI辅助补充）。"""
import pytest
from datasets.rail_dataset import RailDualModalDataset

pytestmark = [pytest.mark.module1, pytest.mark.blackbox]


def frames(dataset):
    return [s["frame_id"] for s in dataset.samples]


def test_count_overrides_ratio(rail_factory):
    dataset = rail_factory(count=10, train_sample_num=4, train_sample_ratio=0.9)
    assert frames(dataset) == ["frame_000", "frame_003", "frame_006", "frame_009"]


def test_requested_count_capped_by_available(rail_factory):
    assert len(rail_factory(count=3, train_sample_num=20)) == 3


def test_ratio_selects_actual_frames(rail_factory):
    dataset = rail_factory(count=10, train_sample_ratio=0.5)
    assert frames(dataset) == ["frame_000", "frame_002", "frame_004", "frame_007", "frame_009"]


def test_random_sampling_reproducible_unique(rail_factory):
    a = rail_factory(count=20, train_sample_num=6, sampling_mode="random", random_seed=7)
    b = rail_factory(count=20, train_sample_num=6, sampling_mode="random", random_seed=7)
    assert frames(a) == frames(b)
    assert len(set(frames(a))) == 6


@pytest.mark.parametrize("mode", ["random", "uniform_time"])
def test_sampled_train_val_partition_is_complete(rail_factory, mode):
    options = dict(count=20, train_sample_num=10, sampling_mode=mode, random_seed=7)
    whole = rail_factory(**options)
    train = rail_factory(**options, train_val_test_split=[0.8, 0.2, 0])
    val = rail_factory(**options, split="val", train_val_test_split=[0.8, 0.2, 0])
    assert len(train) == 8 and len(val) == 2
    assert set(frames(train)).isdisjoint(frames(val))
    assert set(frames(train)) | set(frames(val)) == set(frames(whole))


def test_missing_pair_is_excluded(rail_factory):
    from pathlib import Path
    original = rail_factory(count=2)
    Path(original.samples[0]["depth_path"]).unlink()
    dataset = RailDualModalDataset(original.train_root, original.test_root, 1,
                                   use_patch=False, train_val_test_split=[1, 0, 0])
    assert frames(dataset) == ["frame_001"]


def test_reflectance_and_tif_matching(rail_factory):
    from pathlib import Path
    original = rail_factory()
    rgb = Path(original.samples[0]["rgb_path"])
    depth = Path(original.samples[0]["depth_path"])
    rgb.rename(rgb.with_name("frame_000_reflectance.jpg"))
    depth.rename(depth.with_suffix(".tif"))
    dataset = RailDualModalDataset(original.train_root, original.test_root, 1,
                                   use_patch=False, train_val_test_split=[1, 0, 0])
    assert len(dataset) == 1
    assert dataset[0]["frame_id"] == "frame_000_reflectance"
