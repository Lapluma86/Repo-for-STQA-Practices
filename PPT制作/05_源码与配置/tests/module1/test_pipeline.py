"""最小集成：真实数据加载→批次整理→双模态小模型→蒸馏梯度。
不代表TRD模型精度或完整训练验收。
"""
import pytest
import torch
from torch.utils.data import DataLoader
from utils.losses import loss_l2

pytestmark = [pytest.mark.module1, pytest.mark.integration]


def test_dataset_batch_loss_backward(rail_factory):
    dataset = rail_factory(count=3)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    model = torch.nn.Conv2d(6, 2, kernel_size=1)
    sizes = []
    for batch in loader:
        sizes.append(len(batch["label"]))
        assert batch["intensity"].shape[1:] == (3, 8, 8)
        assert batch["depth"].shape == batch["intensity"].shape
        prediction = model(torch.cat([batch["intensity"], batch["depth"]], dim=1))
        loss = loss_l2([prediction], [torch.zeros_like(prediction)])
        assert torch.isfinite(loss) and loss > 0
        model.zero_grad()
        loss.backward()
        assert torch.isfinite(model.weight.grad).all()
        assert torch.count_nonzero(model.weight.grad) > 0
    assert sizes == [2, 1]
