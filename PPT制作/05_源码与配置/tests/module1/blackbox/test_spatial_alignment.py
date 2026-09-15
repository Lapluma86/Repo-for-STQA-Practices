"""用非均匀图像验证垂直滑窗、水平中心裁剪和GT最近邻缩放。"""
import cv2
import numpy as np
import pytest
import torch
from datasets.rail_dataset import RailDualModalDataset

pytestmark = [pytest.mark.module1, pytest.mark.blackbox]


@pytest.mark.parametrize("size", [4, 8, 16])
def test_rgb_depth_gt_alignment(tmp_path, size):
    rgb_dir = tmp_path / "rail_mvtec/cam1/test/broken"
    depth_dir = tmp_path / "rail_mvtec_depth/cam1/test/broken"
    gt_dir = tmp_path / "rail_mvtec/cam1/ground_truth/broken"
    for folder in (rgb_dir, depth_dir, gt_dir):
        folder.mkdir(parents=True)
    mask = np.zeros((16, 24), np.uint8)
    mask[2:6, 9:12] = 255
    mask[10:14, 12:15] = 255
    rgb = np.repeat(mask[..., None], 3, axis=2)
    depth = np.where(mask > 0, 9, 1).astype(np.uint16)
    assert cv2.imwrite(str(rgb_dir / "frame.jpg"), rgb, [cv2.IMWRITE_JPEG_QUALITY, 100])
    assert cv2.imwrite(str(depth_dir / "frame.tiff"), depth)
    assert cv2.imwrite(str(gt_dir / "frame.png"), mask)
    dataset = RailDualModalDataset(str(tmp_path), str(tmp_path), 1, split="test",
                                   use_patch=True, patch_size=8, patch_stride=8,
                                   img_size=size, depth_norm="log")
    assert len(dataset) == 2
    decoded = cv2.imread(str(rgb_dir / "frame.jpg"))[:, :, ::-1].copy()
    for index in range(2):
        item = dataset[index]
        y = index * 8
        expected_gt = cv2.resize(mask[y:y+8, 8:16], (size, size), interpolation=cv2.INTER_NEAREST) > 0
        expected_depth = cv2.resize(np.log1p(depth[y:y+8, 8:16].astype(np.float32)),
                                   (size, size), interpolation=cv2.INTER_NEAREST)
        torch.testing.assert_close(item["gt"], torch.tensor(expected_gt, dtype=torch.float32))
        torch.testing.assert_close(item["depth"][0], torch.from_numpy(expected_depth))
        source = torch.from_numpy(decoded[y:y+8, 8:16]).permute(2, 0, 1).float() / 255
        expected_rgb = torch.nn.functional.interpolate(source[None], size=(size, size), mode="bilinear", align_corners=False)[0]
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        torch.testing.assert_close(item["intensity"], (expected_rgb-mean)/std)
        assert set(item["gt"].unique().tolist()) <= {0.0, 1.0}
