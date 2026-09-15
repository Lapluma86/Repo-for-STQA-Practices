# -*- coding: utf-8 -*-
"""
白盒测试 - 语句覆盖测试
测试目标：确保所有代码语句至少被执行一次

测试策略：
1. 识别所有可执行语句
2. 设计测试用例覆盖每条语句
3. 使用 pytest-cov 测量覆盖率

作者：手工编写（Phase 1）
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from utils.losses import (
    loss_l2,
    loss_distil,
    loss_distil_p,
    loss_distil_pixel,
    calculate_pixel_similarity,
    calculate_pixel_similarity_f,
    calculate_pixel_similarity_f_2
)
from eval.metrics_engineering import (
    count_trainable_params,
    measure_inference_latency,
    measure_peak_gpu_memory,
    compute_fp_per_image
)


@pytest.mark.whitebox
@pytest.mark.module1
class TestStatementCoverage:
    """语句覆盖测试类"""


    def test_loss_l2_basic(self):
        """
        测试用例ID: TC-SC-001-1
        测试目的: 覆盖 loss_l2 的所有语句
        覆盖语句:
          - line 31: loss_type = torch.nn.MSELoss()
          - line 32: loss = 0.0
          - line 33: for i in range(len(feature_s))
          - line 34: loss_i = loss_type(feature_s[i], feature_t[i])
          - line 35: loss += loss_i
          - line 37: return loss
        """
        # 创建测试特征
        feature_s = [
            torch.randn(2, 64, 32, 32),
            torch.randn(2, 128, 16, 16),
            torch.randn(2, 256, 8, 8)
        ]
        feature_t = [
            torch.randn(2, 64, 32, 32),
            torch.randn(2, 128, 16, 16),
            torch.randn(2, 256, 8, 8)
        ]

        # 执行函数，覆盖所有语句
        loss = loss_l2(feature_s, feature_t)

        # 验证
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0  # MSE 损失非负


    def test_loss_distil_basic(self):
        """
        测试用例ID: TC-SC-002-1
        测试目的: 覆盖 loss_distil 的所有语句
        覆盖语句:
          - line 46: loss_type = torch.nn.CosineSimilarity()
          - line 47: loss = 0.0
          - line 48: for i in range(len(feature_s))
          - line 50-52: loss_i = torch.mean(1 - loss_type(...))
          - line 53: loss += loss_i
          - line 55: return loss
        """
        feature_s = [
            torch.randn(2, 64, 32, 32),
            torch.randn(2, 128, 16, 16)
        ]
        feature_t = [
            torch.randn(2, 64, 32, 32),
            torch.randn(2, 128, 16, 16)
        ]

        loss = loss_distil(feature_s, feature_t)

        assert isinstance(loss, torch.Tensor)
        assert 0 <= loss.item() <= 2.1  # Cosine distance range [0, 2], allow small floating point margin


    def test_loss_distil_p_basic(self):
        """
        测试用例ID: TC-SC-003-1
        测试目的: 覆盖 loss_distil_p 的所有语句
        覆盖语句:
          - line 64: loss_type = torch.nn.CosineSimilarity()
          - line 65: loss = 0.0
          - line 66: for i in range(len(feature_s))
          - line 67: cos = 1 - loss_type(feature_s[i], feature_t[i])
          - line 68: loss_i = torch.mean(cos)
          - line 69: loss += loss_i
          - line 71: return loss
        """
        feature_s = [
            torch.randn(2, 64, 32, 32),
            torch.randn(2, 128, 16, 16),
            torch.randn(2, 256, 8, 8)
        ]
        feature_t = [
            torch.randn(2, 64, 32, 32),
            torch.randn(2, 128, 16, 16),
            torch.randn(2, 256, 8, 8)
        ]

        loss = loss_distil_p(feature_s, feature_t)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0


    def test_calculate_pixel_similarity_f_2_basic(self):
        """
        测试用例ID: TC-SC-007-1
        测试目的: 覆盖 calculate_pixel_similarity_f_2 的所有语句
        """
        f_s = torch.randn(2, 64, 4, 4)
        f_t = torch.randn(2, 64, 4, 4)

        cos_sim = calculate_pixel_similarity_f_2(f_s, f_t)

        assert cos_sim.shape == (2, 16, 16)
        assert cos_sim.min() >= -1.01
        assert cos_sim.max() <= 1.01


    def test_count_trainable_params_frozen_layers(self):
        """
        测试用例ID: TC-SC-008-2
        测试目的: 测试包含冻结层的模型
        """
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Linear(20, 5)
        )

        # 冻结第一层
        for param in model[0].parameters():
            param.requires_grad = False

        total, trainable, ratio = count_trainable_params(model)

        assert total > trainable  # 有冻结参数
        assert 0 < ratio < 1


    def test_compute_fp_per_image_list_input(self):
        """
        测试用例ID: TC-SC-011-4
        测试目的: 测试列表输入（覆盖 np.asarray 转换）
        """
        scores = [0.1, 0.6, 0.7]
        labels = [0, 0, 1]
        threshold = 0.5

        fp_rate = compute_fp_per_image(scores, labels, threshold)

        # scores[0]=0.1<0.5, scores[1]=0.6>0.5
        # FP率 = 1/2 = 0.5
        assert abs(fp_rate - 0.5) < 0.01
