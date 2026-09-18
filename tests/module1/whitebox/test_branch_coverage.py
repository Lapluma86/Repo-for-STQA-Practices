# -*- coding: utf-8 -*-
"""
白盒测试 - 分支覆盖测试
测试目标：确保所有决策分支（if/else）都被执行

测试策略：
1. 识别所有分支点（if/else, for循环, 三元表达式等）
2. 设计测试用例使每个分支都至少执行一次
3. True/False 分支都要覆盖

作者：手工编写（Phase 1）
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from utils.losses import loss_distil_pixel, calculate_pixel_similarity
from eval.metrics_engineering import (
    count_trainable_params,
    measure_inference_latency,
    compute_fp_per_image
)


@pytest.mark.whitebox
@pytest.mark.module1
class TestBranchCoverage:
    """分支覆盖测试类"""


    def test_count_trainable_params_total_greater_than_zero_true(self):
        """
        测试用例ID: TC-BC-001-1
        分支: total > 0 (True)
        测试目的: 覆盖 ratio = trainable / total 分支
        """
        model = nn.Linear(10, 5)

        total, trainable, ratio = count_trainable_params(model)

        # total > 0，执行除法
        assert total > 0
        assert ratio == trainable / total

    def test_count_trainable_params_total_greater_than_zero_false(self):
        """
        测试用例ID: TC-BC-001-2
        分支: total > 0 (False)
        测试目的: 覆盖 ratio = 0.0 分支
        """
        model = nn.Sequential()  # 空模型

        total, trainable, ratio = count_trainable_params(model)

        # total == 0，执行 else 分支
        assert total == 0
        assert ratio == 0.0


    def test_compute_fp_per_image_n_normal_zero_true(self):
        """
        测试用例ID: TC-BC-004-1
        分支: n_normal == 0 (True)
        测试目的: 没有正常样本时返回 0.0
        """
        scores = np.array([0.8, 0.9, 0.7])
        labels = np.array([1, 1, 1])  # 全是异常
        threshold = 0.5

        fp_rate = compute_fp_per_image(scores, labels, threshold)

        # 执行 return 0.0 分支
        assert fp_rate == 0.0

    def test_compute_fp_per_image_n_normal_zero_false(self):
        """
        测试用例ID: TC-BC-004-2
        分支: n_normal == 0 (False)
        测试目的: 有正常样本时计算误报率
        """
        scores = np.array([0.6, 0.3, 0.7])
        labels = np.array([0, 0, 1])  # 有正常样本
        threshold = 0.5

        fp_rate = compute_fp_per_image(scores, labels, threshold)

        # 执行 return float(...) / n_normal 分支
        # 正常样本: [0.6, 0.3], >0.5的: [0.6]
        # FP率 = 1/2 = 0.5
        assert abs(fp_rate - 0.5) < 0.01


    def test_loss_distil_pixel_all_branches(self):
        """
        测试用例ID: TC-BC-005-4
        分支: 完整测试所有分支
        测试目的: 确保三个尺度的不同分支都被覆盖
        """
        feature_s = [
            torch.randn(1, 64, 16, 16),   # i=0: pool
            torch.randn(1, 128, 8, 8),    # i=1: pool
            torch.randn(1, 256, 4, 4)     # i=2: no pool
        ]
        feature_t = [
            torch.randn(1, 64, 16, 16),
            torch.randn(1, 128, 8, 8),
            torch.randn(1, 256, 4, 4)
        ]

        loss = loss_distil_pixel(feature_s, feature_t)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
