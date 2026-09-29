"""从精简前的采样、边界和等价类程序恢复的数据规模用例。

只保留能核对帧号、属性或文件配对的检查。构造期 ValueError 仍由既有 16 项覆盖。
"""
import pytest
from datasets.rail_dataset import RailDualModalDataset

pytestmark = [pytest.mark.module2, pytest.mark.ai_assisted]


def frames(dataset):
    return [sample["frame_id"] for sample in dataset.samples]


def test_ratio_selects_actual_frames(rail_factory):
    dataset = rail_factory(count=10, train_sample_ratio=0.5)
    assert frames(dataset) == ["frame_000", "frame_002", "frame_004", "frame_007", "frame_009"]


def test_requested_count_capped_by_available(rail_factory):
    assert len(rail_factory(count=3, train_sample_num=20)) == 3


@pytest.mark.parametrize("mode", ["uniform_time", "random"])
def test_train_val_partition_complete(rail_factory, mode):
    options = dict(count=20, train_sample_num=10, sampling_mode=mode, random_seed=7)
    whole = rail_factory(**options)
    train = rail_factory(**options, train_val_test_split=[0.8, 0.2, 0])
    val = rail_factory(**options, split="val", train_val_test_split=[0.8, 0.2, 0])
    assert len(train) == 8 and len(val) == 2
    assert set(frames(train)).isdisjoint(frames(val))
    assert set(frames(train)) | set(frames(val)) == set(frames(whole))


def test_missing_pair_is_excluded(rail_factory):
    original = rail_factory(count=2)
    from pathlib import Path
    Path(original.samples[0]["depth_path"]).unlink()
    dataset = RailDualModalDataset(
        original.train_root, original.test_root, 1,
        use_patch=False, train_val_test_split=[1, 0, 0],
    )
    assert frames(dataset) == ["frame_001"]


def test_reflectance_and_tif_matching(rail_factory):
    from pathlib import Path
    original = rail_factory()
    rgb = Path(original.samples[0]["rgb_path"])
    depth = Path(original.samples[0]["depth_path"])
    rgb.rename(rgb.with_name("frame_000_reflectance.jpg"))
    depth.rename(depth.with_suffix(".tif"))
    dataset = RailDualModalDataset(
        original.train_root, original.test_root, 1,
        use_patch=False, train_val_test_split=[1, 0, 0],
    )
    assert len(dataset) == 1
    assert dataset[0]["frame_id"] == "frame_000_reflectance"


@pytest.mark.parametrize("img_size", [2, 32, 128])
def test_img_size_representatives_are_stored(rail_factory, img_size):
    dataset = rail_factory(img_size=img_size)
    assert dataset.img_size == img_size
    assert dataset[0]["intensity"].shape[-1] == img_size
    assert dataset[0]["depth"].shape[-1] == img_size


@pytest.mark.parametrize("view_id", [1, 8])
def test_view_id_endpoints_are_stored(rail_factory, view_id):
    dataset = rail_factory(view_id=view_id)
    assert dataset.view_id == view_id
    assert dataset[0]["view_id"] == view_id


@pytest.mark.parametrize("depth_norm", ["zscore", "log"])
def test_depth_norm_representatives_stay_finite(rail_factory, depth_norm):
    dataset = rail_factory(depth_norm=depth_norm)
    assert dataset.depth_norm == depth_norm
    depth = dataset[0]["depth"]
    assert depth.shape[0] == 3
    assert torch_isfinite(depth)


def torch_isfinite(tensor):
    import torch
    return bool(torch.isfinite(tensor).all())


def test_split_val_keeps_disjoint_frames(rail_factory):
    train = rail_factory(count=10, train_val_test_split=[0.8, 0.2, 0])
    val = rail_factory(count=10, split="val", train_val_test_split=[0.8, 0.2, 0])
    assert train.split == "train" and val.split == "val"
    assert len(train) == 8 and len(val) == 2
    assert set(frames(train)).isdisjoint(frames(val))


def test_patch_stride_equal_to_size_counts_two_patches(rail_factory):
    dataset = rail_factory(use_patch=True, patch_size=8, patch_stride=8)
    assert dataset.num_patches == 2
    assert [dataset[i]["patch_idx"] for i in range(2)] == [0, 1]


def test_empty_test_split_is_empty(rail_factory):
    dataset = rail_factory(split="test")
    assert dataset.split == "test"
    assert len(dataset) == 0
