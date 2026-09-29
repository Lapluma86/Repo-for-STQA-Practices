"""从精简前的状态与判定表程序恢复的读取行为。

预加载一致性只用 minmax 和 log。z-score 切块差异仍由原有三项属性测试暴露。
"""
import cv2
import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.module2, pytest.mark.ai_assisted]


def test_repeated_read_is_stable(rail_factory):
    dataset = rail_factory(depth_norm="minmax")
    first = dataset[0]
    for _ in range(2):
        again = dataset[0]
        for key in ("intensity", "depth", "gt"):
            torch.testing.assert_close(first[key], again[key])


def test_patch_configuration_changes_length(rail_factory):
    full = rail_factory()
    patches = rail_factory(use_patch=True, patch_size=8, patch_stride=8)
    assert len(full) == 1
    assert len(patches) == 2
    assert patches[0]["depth"].shape == (3, 8, 8)


def test_minmax_preload_matches_lazy_patches(rail_factory):
    lazy = rail_factory(use_patch=True, depth_norm="minmax", patch_size=8, patch_stride=8)
    cached = rail_factory(use_patch=True, depth_norm="minmax", patch_size=8, patch_stride=8, preload=True)
    for index in (1, 0):
        torch.testing.assert_close(lazy[index]["depth"], cached[index]["depth"])


def test_log_patch_matches_cropped_log1p(rail_factory):
    dataset = rail_factory(use_patch=True, depth_norm="log", patch_size=8, patch_stride=8)
    full = np.arange(1, 385, dtype=np.float32).reshape(16, 24)
    expected = np.log1p(full[0:8, 8:16])
    torch.testing.assert_close(dataset[0]["depth"][0], torch.from_numpy(expected))


def test_zscore_full_frame_matches_hand_calculation(rail_factory):
    dataset = rail_factory(use_patch=False, depth_norm="zscore", img_size=8)
    full = np.arange(1, 385, dtype=np.float32).reshape(16, 24)
    normalized = (full - full.mean()) / (full.std() + 1e-6)
    expected = cv2.resize(normalized, (8, 8), interpolation=cv2.INTER_NEAREST)
    torch.testing.assert_close(dataset[0]["depth"][0], torch.from_numpy(expected))


def test_train_and_val_frames_stay_disjoint(rail_factory):
    train = rail_factory(count=10, train_val_test_split=[0.8, 0.2, 0])
    val = rail_factory(count=10, split="val", train_val_test_split=[0.8, 0.2, 0])
    assert len(train) == 8 and len(val) == 2
    assert {sample["frame_id"] for sample in train.samples}.isdisjoint(
        sample["frame_id"] for sample in val.samples
    )


def test_index_error_does_not_block_the_valid_item(rail_factory):
    dataset = rail_factory(depth_norm="log")
    with pytest.raises(IndexError):
        dataset[1]
    item = dataset[0]
    assert item["frame_id"] == "frame_000"
    assert torch.isfinite(item["depth"]).all()


def test_test_split_reads_good_and_broken(rail_factory, tmp_path):
    for category, label_fill in (("good", 0), ("broken", 255)):
        rgb = tmp_path / "test/rail_mvtec/cam1/test" / category
        depth = tmp_path / "test/rail_mvtec_depth/cam1/test" / category
        rgb.mkdir(parents=True)
        depth.mkdir(parents=True)
        assert cv2.imwrite(str(rgb / "frame.jpg"), np.full((16, 24, 3), 40, np.uint8))
        assert cv2.imwrite(str(depth / "frame.tiff"), np.full((16, 24), 3, np.uint16))
        if category == "broken":
            gt_dir = tmp_path / "test/rail_mvtec/cam1/ground_truth/broken"
            gt_dir.mkdir(parents=True)
            assert cv2.imwrite(str(gt_dir / "frame.png"), np.full((16, 24), label_fill, np.uint8))
    dataset = rail_factory(split="test", depth_norm="log")
    assert len(dataset) == 2
    by_label = {dataset[i]["label"]: dataset[i] for i in range(2)}
    assert set(by_label) == {0, 1}
    assert torch.count_nonzero(by_label[0]["gt"]) == 0
    assert torch.all(by_label[1]["gt"] == 1)
