# -*- coding: utf-8 -*-
"""
白盒测试 - 条件覆盖测试
测试目标：确保复合条件中的每个子条件都至少取True和False一次

测试策略：
1. 识别所有复合条件（使用 and/or 的条件）
2. 设计测试用例使每个子条件都取True和False
3. 记录条件组合的真值表

作者：手工编写（Phase 1）
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from utils.losses import loss_distil_pixel
from eval.metrics_engineering import count_trainable_params


@pytest.mark.whitebox
@pytest.mark.module1
class TestConditionCoverage:
    """条件覆盖测试类"""


    def test_condition_i_equals_0_true_i_equals_1_false(self):
        """
        测试用例ID: TC-CC-001-1
        复合条件: i == 0 or i == 1
        子条件1 (i == 0): True
        子条件2 (i == 1): False
        测试目的: 覆盖 i==0 为True的情况
        """
        # i=0 时，i==0为True，i==1为False
        feature_s = [torch.randn(1, 64, 16, 16)]  # 只有 i=0
        feature_t = [torch.randn(1, 64, 16, 16)]

        loss = loss_distil_pixel(feature_s, feature_t)

        assert isinstance(loss, torch.Tensor)

    def test_condition_i_equals_0_false_i_equals_1_true(self):
        """
        测试用例ID: TC-CC-001-2
        复合条件: i == 0 or i == 1
        子条件1 (i == 0): False
        子条件2 (i == 1): True
        测试目的: 覆盖 i==1 为True的情况
        """
        # 通过使用2个元素，让循环执行到 i=1
        # i=1 时，i==0为False，i==1为True
        feature_s = [
            torch.randn(1, 64, 16, 16),   # i=0
            torch.randn(1, 128, 8, 8)     # i=1
        ]
        feature_t = [
            torch.randn(1, 64, 16, 16),
            torch.randn(1, 128, 8, 8)
        ]

        loss = loss_distil_pixel(feature_s, feature_t)

        assert isinstance(loss, torch.Tensor)

    def test_condition_i_equals_0_false_i_equals_1_false(self):
        """
        测试用例ID: TC-CC-001-3
        复合条件: i == 0 or i == 1
        子条件1 (i == 0): False
        子条件2 (i == 1): False
        测试目的: 覆盖两个子条件都为False的情况
        """
        # i=2 时，i==0为False，i==1为False
        feature_s = [
            torch.randn(1, 64, 16, 16),
            torch.randn(1, 128, 8, 8),
            torch.randn(1, 256, 4, 4)     # i=2
        ]
        feature_t = [
            torch.randn(1, 64, 16, 16),
            torch.randn(1, 128, 8, 8),
            torch.randn(1, 256, 4, 4)
        ]

        loss = loss_distil_pixel(feature_s, feature_t)

        assert isinstance(loss, torch.Tensor)


    def test_condition_isinstance_tuple_true(self):
        """
        测试用例ID: TC-CC-003-1
        复合条件: isinstance(sample_input, (tuple, list))
        子条件1 (isinstance tuple): True
        子条件2 (isinstance list): 不需要判断（短路）
        测试目的: 覆盖输入为tuple的情况
        """
        from eval.metrics_engineering import measure_inference_latency

        model = nn.Linear(10, 5)
        sample_input = (torch.randn(1, 10),)  # tuple
        device = torch.device("cpu")

        latency = measure_inference_latency(
            model, sample_input, device,
            n_warmup=2, n_run=5
        )

        assert latency > 0


    def test_condition_isinstance_both_false(self):
        """
        测试用例ID: TC-CC-003-3
        复合条件: isinstance(sample_input, (tuple, list))
        子条件1 (isinstance tuple): False
        子条件2 (isinstance list): False
        测试目的: 覆盖输入既不是tuple也不是list的情况
        """
        from eval.metrics_engineering import measure_inference_latency

        model = nn.Linear(10, 5)
        sample_input = torch.randn(1, 10)  # 既不是tuple也不是list
        device = torch.device("cpu")

        # 断言失败
        with pytest.raises(AssertionError):
            measure_inference_latency(model, sample_input, device)
