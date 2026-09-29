"""从精简前的损失数值和工程指标程序恢复的 CPU 用例。

余弦同向、反向和误报率使用可手算的预期。非法 warmup 沿用原始断言。
"""
import numpy as np
import pytest
import torch
from eval.metrics_engineering import (
    compute_fp_per_image,
    count_trainable_params,
    measure_inference_latency,
)
from utils.losses import loss_distil, loss_distil_p, loss_l2

pytestmark = [pytest.mark.module2, pytest.mark.ai_assisted]


@pytest.mark.parametrize("loss_fn", [loss_distil, loss_distil_p])
@pytest.mark.parametrize("sign, expected", [(1, 0.0), (-1, 2.0)])
def test_cosine_same_and_opposite(loss_fn, sign, expected):
    feature = torch.tensor([[[[1.0]], [[2.0]]]])
    assert loss_fn([feature], [feature * sign]).item() == pytest.approx(expected, abs=1e-6)


def test_l2_single_scale_is_four():
    student = torch.tensor([[[[2.0]]]], requires_grad=True)
    value = loss_l2([student], [torch.zeros_like(student)])
    assert value.item() == pytest.approx(4.0)
    value.backward()
    torch.testing.assert_close(student.grad, torch.tensor([[[[4.0]]]]))


def test_distil_identical_features_are_zero():
    feature = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    assert loss_distil([feature], [feature]).item() == pytest.approx(0.0, abs=1e-6)


def test_fp_rate_zero_when_normals_are_below_threshold():
    scores = np.array([0.1, 0.3, 0.7, 0.9, 0.2])
    labels = np.array([0, 0, 1, 1, 0])
    assert compute_fp_per_image(scores, labels, 0.5) == 0.0


def test_fp_rate_two_of_three_normals():
    scores = np.array([0.1, 0.6, 0.7, 0.9, 0.8])
    labels = np.array([0, 0, 1, 1, 0])
    assert compute_fp_per_image(scores, labels, 0.5) == pytest.approx(2 / 3)


def test_fp_rate_without_normal_samples_is_zero():
    assert compute_fp_per_image(np.array([0.7, 0.8, 0.9]), np.array([1, 1, 1]), 0.5) == 0.0


def test_empty_model_has_zero_parameters():
    total, trainable, ratio = count_trainable_params(torch.nn.Identity())
    assert (total, trainable, ratio) == (0, 0, 0.0)


def test_negative_warmup_is_rejected():
    with pytest.raises(ValueError, match="n_warmup"):
        measure_inference_latency(
            torch.nn.Identity(), (torch.ones(1),), torch.device("cpu"), n_warmup=-1, n_run=1,
        )
