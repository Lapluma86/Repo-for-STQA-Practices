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
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.losses import loss_distil_pixel, calculate_pixel_similarity
from eval.metrics_engineering import (
    count_trainable_params,
    measure_inference_latency,
    measure_peak_gpu_memory,
    compute_fp_per_image
)


@pytest.mark.whitebox
@pytest.mark.phase1
class TestBranchCoverage:
    """分支覆盖测试类"""

    # ===== 测试用例 TC-BC-001: count_trainable_params 分支覆盖 =====

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

    # ===== 测试用例 TC-BC-002: measure_inference_latency 分支覆盖 =====

    def test_measure_inference_latency_cuda_false(self):
        """
        测试用例ID: TC-BC-002-1
        分支: use_cuda = False
        测试目的: CPU 设备分支，不执行 torch.cuda.synchronize()
        """
        model = nn.Linear(10, 5)
        sample_input = (torch.randn(1, 10),)
        device = torch.device("cpu")

        latency = measure_inference_latency(
            model, sample_input, device,
            n_warmup=2, n_run=5
        )

        assert latency > 0

    def test_measure_inference_latency_cuda_true(self):
        """
        测试用例ID: TC-BC-002-2
        分支: use_cuda = True
        测试目的: CUDA 设备分支，执行 torch.cuda.synchronize()
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        model = nn.Linear(10, 5).cuda()
        sample_input = (torch.randn(1, 10).cuda(),)
        device = torch.device("cuda")

        latency = measure_inference_latency(
            model, sample_input, device,
            n_warmup=2, n_run=5
        )

        assert latency > 0

    # ===== 测试用例 TC-BC-003: measure_peak_gpu_memory 分支覆盖 =====

    def test_measure_peak_gpu_memory_not_cuda_true(self):
        """
        测试用例ID: TC-BC-003-1
        分支: device.type != "cuda" (True)
        测试目的: 非 CUDA 设备返回 0.0
        """
        device = torch.device("cpu")

        memory = measure_peak_gpu_memory(device)

        # 执行 return 0.0 分支
        assert memory == 0.0

    def test_measure_peak_gpu_memory_not_cuda_false(self):
        """
        测试用例ID: TC-BC-003-2
        分支: device.type != "cuda" (False)
        测试目的: CUDA 设备返回实际内存使用
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = torch.device("cuda")
        torch.cuda.reset_peak_memory_stats(device)

        # 分配内存
        _ = torch.randn(100, 100).cuda()

        memory = measure_peak_gpu_memory(device)

        # 执行 return torch.cuda.max_memory_allocated(...) 分支
        assert memory > 0

    # ===== 测试用例 TC-BC-004: compute_fp_per_image 分支覆盖 =====

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

    # ===== 测试用例 TC-BC-005: loss_distil_pixel 分支覆盖 =====

    def test_loss_distil_pixel_pool_branch_i_equals_0(self):
        """
        测试用例ID: TC-BC-005-1
        分支: i == 0 (True)
        测试目的: 第0个尺度执行 pool_s[0] 和 pool_t[0]
        """
        # 确保至少有一个尺度，且测试 i=0
        feature_s = [torch.randn(1, 64, 16, 16)]
        feature_t = [torch.randn(1, 64, 16, 16)]

        loss = loss_distil_pixel(feature_s, feature_t)

        # i=0 时应该执行池化
        assert isinstance(loss, torch.Tensor)

    def test_loss_distil_pixel_pool_branch_i_equals_1(self):
        """
        测试用例ID: TC-BC-005-2
        分支: i == 1 (True)
        测试目的: 第1个尺度执行 pool_s[1] 和 pool_t[1]
        """
        feature_s = [
            torch.randn(1, 64, 16, 16),  # i=0
            torch.randn(1, 128, 8, 8)    # i=1
        ]
        feature_t = [
            torch.randn(1, 64, 16, 16),
            torch.randn(1, 128, 8, 8)
        ]

        loss = loss_distil_pixel(feature_s, feature_t)

        # i=1 时应该执行池化
        assert isinstance(loss, torch.Tensor)

    def test_loss_distil_pixel_pool_branch_i_not_0_or_1(self):
        """
        测试用例ID: TC-BC-005-3
        分支: i == 0 or i == 1 (False)
        测试目的: 第2个及以后的尺度不执行池化
        """
        feature_s = [
            torch.randn(1, 64, 16, 16),   # i=0
            torch.randn(1, 128, 8, 8),    # i=1
            torch.randn(1, 256, 4, 4)     # i=2
        ]
        feature_t = [
            torch.randn(1, 64, 16, 16),
            torch.randn(1, 128, 8, 8),
            torch.randn(1, 256, 4, 4)
        ]

        loss = loss_distil_pixel(feature_s, feature_t)

        # i=2 时不应该执行池化
        assert isinstance(loss, torch.Tensor)

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

    # ===== 测试用例 TC-BC-006: 循环分支覆盖 =====

    def test_loop_branch_zero_iterations(self):
        """
        测试用例ID: TC-BC-006-1
        分支: 循环0次（空列表）
        测试目的: 测试循环体不执行的情况
        """
        from utils.losses import loss_l2

        feature_s = []
        feature_t = []

        loss = loss_l2(feature_s, feature_t)

        # 循环0次，loss保持初始值0.0
        assert loss == 0.0

    def test_loop_branch_one_iteration(self):
        """
        测试用例ID: TC-BC-006-2
        分支: 循环1次
        测试目的: 测试循环体执行一次
        """
        from utils.losses import loss_l2

        feature_s = [torch.randn(2, 64, 32, 32)]
        feature_t = [torch.randn(2, 64, 32, 32)]

        loss = loss_l2(feature_s, feature_t)

        assert isinstance(loss, torch.Tensor)

    def test_loop_branch_multiple_iterations(self):
        """
        测试用例ID: TC-BC-006-3
        分支: 循环多次
        测试目的: 测试循环体执行多次
        """
        from utils.losses import loss_l2

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

        loss = loss_l2(feature_s, feature_t)

        assert isinstance(loss, torch.Tensor)

    # ===== 测试用例 TC-BC-007: calculate_pixel_similarity 批次循环分支 =====

    def test_calculate_pixel_similarity_batch_loop_single(self):
        """
        测试用例ID: TC-BC-007-1
        分支: batch循环执行1次
        测试目的: batch_size=1时的分支
        """
        f_s = torch.randn(1, 32, 4, 4)
        f_t = torch.randn(1, 32, 4, 4)

        cos_sim_s, cos_sim_t = calculate_pixel_similarity(f_s, f_t)

        assert cos_sim_s.shape[0] == 1

    def test_calculate_pixel_similarity_batch_loop_multiple(self):
        """
        测试用例ID: TC-BC-007-2
        分支: batch循环执行多次
        测试目的: batch_size>1时的分支
        """
        f_s = torch.randn(4, 32, 4, 4)
        f_t = torch.randn(4, 32, 4, 4)

        cos_sim_s, cos_sim_t = calculate_pixel_similarity(f_s, f_t)

        assert cos_sim_s.shape[0] == 4

    # ===== 测试用例 TC-BC-008: 组合分支测试 =====

    def test_combined_branches_all_true(self):
        """
        测试用例ID: TC-BC-008-1
        测试目的: 测试多个条件都为True的组合
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # use_cuda = True, total > 0
        model = nn.Linear(10, 5).cuda()
        sample_input = (torch.randn(1, 10).cuda(),)
        device = torch.device("cuda")

        latency = measure_inference_latency(
            model, sample_input, device,
            n_warmup=2, n_run=5
        )

        total, trainable, ratio = count_trainable_params(model)

        assert latency > 0
        assert total > 0
        assert ratio > 0

    def test_combined_branches_mixed(self):
        """
        测试用例ID: TC-BC-008-2
        测试目的: 测试条件混合的组合（部分True，部分False）
        """
        # use_cuda = False (CPU), total > 0 = True (有参数)
        model = nn.Linear(10, 5)
        sample_input = (torch.randn(1, 10),)
        device = torch.device("cpu")

        latency = measure_inference_latency(
            model, sample_input, device,
            n_warmup=2, n_run=5
        )

        total, trainable, ratio = count_trainable_params(model)

        assert latency > 0
        assert total > 0

    def test_combined_branches_all_false(self):
        """
        测试用例ID: TC-BC-008-3
        测试目的: 测试多个条件都为False的组合
        """
        # use_cuda = False (CPU), n_normal == 0 (无正常样本)
        scores = np.array([0.8, 0.9])
        labels = np.array([1, 1])  # 全异常
        threshold = 0.5

        fp_rate = compute_fp_per_image(scores, labels, threshold)

        device = torch.device("cpu")
        memory = measure_peak_gpu_memory(device)

        assert fp_rate == 0.0  # n_normal == 0 分支
        assert memory == 0.0   # not cuda 分支

    # ===== 测试用例 TC-BC-009: 嵌套分支覆盖 =====

    def test_nested_branch_outer_true_inner_true(self):
        """
        测试用例ID: TC-BC-009-1
        分支: 外层True, 内层True
        测试目的: loss_distil_pixel 中 i==0 时执行池化
        """
        feature_s = [torch.randn(1, 64, 16, 16)]
        feature_t = [torch.randn(1, 64, 16, 16)]

        loss = loss_distil_pixel(feature_s, feature_t)

        # 外层循环执行，内层 i==0 条件为True
        assert isinstance(loss, torch.Tensor)

    def test_nested_branch_outer_true_inner_false(self):
        """
        测试用例ID: TC-BC-009-2
        分支: 外层True, 内层False
        测试目的: loss_distil_pixel 中 i==2 时不执行池化
        """
        feature_s = [
            torch.randn(1, 64, 16, 16),
            torch.randn(1, 128, 8, 8),
            torch.randn(1, 256, 4, 4)  # i=2, 不池化
        ]
        feature_t = [
            torch.randn(1, 64, 16, 16),
            torch.randn(1, 128, 8, 8),
            torch.randn(1, 256, 4, 4)
        ]

        loss = loss_distil_pixel(feature_s, feature_t)

        # 外层循环执行，内层 i==0 or i==1 条件为False
        assert isinstance(loss, torch.Tensor)

    # ===== 测试用例 TC-BC-010: 断言分支 =====

    def test_assertion_branch_true(self):
        """
        测试用例ID: TC-BC-010-1
        分支: 断言通过
        测试目的: measure_inference_latency 的 assert 为True
        """
        model = nn.Linear(10, 5)
        sample_input = (torch.randn(1, 10),)  # 正确的tuple
        device = torch.device("cpu")

        latency = measure_inference_latency(model, sample_input, device)

        # 断言通过，继续执行
        assert latency > 0

    def test_assertion_branch_false(self):
        """
        测试用例ID: TC-BC-010-2
        分支: 断言失败
        测试目的: measure_inference_latency 的 assert 为False
        """
        model = nn.Linear(10, 5)
        sample_input = torch.randn(1, 10)  # 错误：不是tuple
        device = torch.device("cpu")

        # 断言失败，抛出 AssertionError
        with pytest.raises(AssertionError):
            measure_inference_latency(model, sample_input, device)
