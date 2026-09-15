"""数值预期验证：避免仅检查类型/非负而接受恒定返回0的错误实现。"""
import pytest
import torch
from utils.losses import loss_l2, loss_distil, loss_distil_p

pytestmark = [pytest.mark.module1, pytest.mark.whitebox]


def test_l2_multiscale_value_and_gradient():
    first = torch.tensor([[[[1.0, 3.0]]]], requires_grad=True)
    second = torch.tensor([[[[2.0]]]], requires_grad=True)
    value = loss_l2([first, second], [torch.zeros_like(first), torch.zeros_like(second)])
    assert value.item() == pytest.approx(9.0)  # (1+9)/2 + 4
    value.backward()
    torch.testing.assert_close(first.grad, torch.tensor([[[[1.0, 3.0]]]]))
    torch.testing.assert_close(second.grad, torch.tensor([[[[4.0]]]]))


@pytest.mark.parametrize("loss_fn", [loss_distil, loss_distil_p])
@pytest.mark.parametrize("sign, expected", [(1, 0.0), (-1, 2.0)])
def test_cosine_same_and_opposite(loss_fn, sign, expected):
    feature = torch.tensor([[[[1.0]], [[2.0]]]])
    assert loss_fn([feature], [feature * sign]).item() == pytest.approx(expected, abs=1e-6)
