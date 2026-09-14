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
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.losses import loss_distil_pixel
from eval.metrics_engineering import count_trainable_params


@pytest.mark.whitebox
@pytest.mark.phase1
class TestConditionCoverage:
    """条件覆盖测试类"""

    # ===== 测试用例 TC-CC-001: loss_distil_pixel 复合条件 (i == 0 or i == 1) =====

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

    def test_condition_i_equals_0_true_i_equals_1_true(self):
        """
        测试用例ID: TC-CC-001-4
        复合条件: i == 0 or i == 1
        说明: 在单次执行中，i不能同时为0和1，但可以通过多次迭代覆盖
        测试目的: 通过循环覆盖两个子条件都为True的情况（不同迭代）
        """
        # 包含 i=0 和 i=1 的迭代
        feature_s = [
            torch.randn(1, 64, 16, 16),   # i=0: 第一个条件True
            torch.randn(1, 128, 8, 8)     # i=1: 第二个条件True
        ]
        feature_t = [
            torch.randn(1, 64, 16, 16),
            torch.randn(1, 128, 8, 8)
        ]

        loss = loss_distil_pixel(feature_s, feature_t)

        # 覆盖了 i==0 为True 和 i==1 为True（不同迭代）
        assert isinstance(loss, torch.Tensor)

    # ===== 测试用例 TC-CC-002: count_trainable_params 三元条件 (total > 0) =====

    def test_condition_total_greater_than_zero_true(self):
        """
        测试用例ID: TC-CC-002-1
        条件: total > 0
        条件值: True
        测试目的: 覆盖 total > 0 为True，执行除法
        """
        model = nn.Linear(10, 5)

        total, trainable, ratio = count_trainable_params(model)

        assert total > 0
        # 验证执行了 trainable / total
        assert ratio == trainable / total

    def test_condition_total_greater_than_zero_false(self):
        """
        测试用例ID: TC-CC-002-2
        条件: total > 0
        条件值: False
        测试目的: 覆盖 total > 0 为False，返回0.0
        """
        model = nn.Sequential()  # 空模型，total=0

        total, trainable, ratio = count_trainable_params(model)

        assert total == 0
        # 验证执行了 else 分支，返回 0.0
        assert ratio == 0.0

    # ===== 测试用例 TC-CC-003: isinstance 复合条件 =====

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

    def test_condition_isinstance_list_true(self):
        """
        测试用例ID: TC-CC-003-2
        复合条件: isinstance(sample_input, (tuple, list))
        子条件1 (isinstance tuple): False
        子条件2 (isinstance list): True
        测试目的: 覆盖输入为list的情况
        """
        from eval.metrics_engineering import measure_inference_latency

        model = nn.Linear(10, 5)
        sample_input = [torch.randn(1, 10)]  # list
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

    # ===== 测试用例 TC-CC-004: 不等式条件覆盖 =====

    def test_condition_device_type_equals_cuda_true(self):
        """
        测试用例ID: TC-CC-004-1
        条件: device.type == "cuda"
        条件值: True
        测试目的: 覆盖 CUDA 设备
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from eval.metrics_engineering import measure_peak_gpu_memory

        device = torch.device("cuda")
        torch.cuda.reset_peak_memory_stats(device)
        _ = torch.randn(100, 100).cuda()

        memory = measure_peak_gpu_memory(device)

        assert memory > 0

    def test_condition_device_type_equals_cuda_false(self):
        """
        测试用例ID: TC-CC-004-2
        条件: device.type == "cuda"
        条件值: False
        测试目的: 覆盖非 CUDA 设备
        """
        from eval.metrics_engineering import measure_peak_gpu_memory

        device = torch.device("cpu")

        memory = measure_peak_gpu_memory(device)

        assert memory == 0.0

    def test_condition_device_type_not_cuda_true(self):
        """
        测试用例ID: TC-CC-004-3
        条件: device.type != "cuda"
        条件值: True
        测试目的: 覆盖不等于判断为True
        """
        from eval.metrics_engineering import measure_peak_gpu_memory

        device = torch.device("cpu")

        memory = measure_peak_gpu_memory(device)

        # device.type != "cuda" 为True
        assert memory == 0.0

    def test_condition_device_type_not_cuda_false(self):
        """
        测试用例ID: TC-CC-004-4
        条件: device.type != "cuda"
        条件值: False
        测试目的: 覆盖不等于判断为False
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from eval.metrics_engineering import measure_peak_gpu_memory

        device = torch.device("cuda")

        # device.type != "cuda" 为False，执行else分支
        memory = measure_peak_gpu_memory(device)

        assert isinstance(memory, float)

    # ===== 测试用例 TC-CC-005: 比较运算符条件覆盖 =====

    def test_condition_n_normal_equals_zero_true(self):
        """
        测试用例ID: TC-CC-005-1
        条件: n_normal == 0
        条件值: True
        测试目的: 正常样本数为0
        """
        from eval.metrics_engineering import compute_fp_per_image

        scores = np.array([0.8, 0.9])
        labels = np.array([1, 1])  # 全异常
        threshold = 0.5

        fp_rate = compute_fp_per_image(scores, labels, threshold)

        assert fp_rate == 0.0

    def test_condition_n_normal_equals_zero_false(self):
        """
        测试用例ID: TC-CC-005-2
        条件: n_normal == 0
        条件值: False
        测试目的: 正常样本数不为0
        """
        from eval.metrics_engineering import compute_fp_per_image

        scores = np.array([0.6, 0.3])
        labels = np.array([0, 0])  # 有正常样本
        threshold = 0.5

        fp_rate = compute_fp_per_image(scores, labels, threshold)

        # n_normal > 0，执行计算
        assert isinstance(fp_rate, float)

    def test_condition_labels_equals_zero_true_and_false(self):
        """
        测试用例ID: TC-CC-005-3
        条件: labels == 0 (数组条件)
        测试目的: 覆盖数组元素既有True也有False的情况
        """
        from eval.metrics_engineering import compute_fp_per_image

        scores = np.array([0.6, 0.3, 0.8])
        labels = np.array([0, 0, 1])  # 前两个True，最后一个False
        threshold = 0.5

        fp_rate = compute_fp_per_image(scores, labels, threshold)

        # labels==0 产生 [True, True, False]
        assert isinstance(fp_rate, float)

    def test_condition_scores_greater_than_threshold_true_and_false(self):
        """
        测试用例ID: TC-CC-005-4
        条件: scores > threshold (数组条件)
        测试目的: 覆盖数组元素既有True也有False的情况
        """
        from eval.metrics_engineering import compute_fp_per_image

        scores = np.array([0.3, 0.7, 0.9])  # 第一个False，后两个True
        labels = np.array([0, 0, 0])
        threshold = 0.5

        fp_rate = compute_fp_per_image(scores, labels, threshold)

        # scores > threshold 产生 [False, True, True]
        # 正常样本中，大于阈值的有2个，FP率 = 2/3
        assert abs(fp_rate - 2/3) < 0.01

    # ===== 测试用例 TC-CC-006: 条件真值表完整性测试 =====

    def test_condition_truth_table_complete(self):
        """
        测试用例ID: TC-CC-006-1
        测试目的: 验证条件覆盖的完整性

        真值表：
        条件A (total > 0) | 条件B (device != cuda) | 测试用例
        -----------------|----------------------|--------
        True             | True                 | TC-CC-006-1a
        True             | False                | TC-CC-006-1b
        False            | True                 | TC-CC-006-1c
        False            | False                | (不可能：空模型+CUDA)
        """
        # TC-CC-006-1a: True, True
        model = nn.Linear(10, 5)  # total > 0: True
        device = torch.device("cpu")  # device != "cuda": True

        total, trainable, ratio = count_trainable_params(model)
        assert total > 0 and ratio > 0

        from eval.metrics_engineering import measure_peak_gpu_memory
        memory = measure_peak_gpu_memory(device)
        assert memory == 0.0

    def test_condition_truth_table_true_false(self):
        """
        测试用例ID: TC-CC-006-1b
        真值表行: True, False
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        model = nn.Linear(10, 5)  # total > 0: True
        device = torch.device("cuda")  # device != "cuda": False

        total, trainable, ratio = count_trainable_params(model)
        assert total > 0

        from eval.metrics_engineering import measure_peak_gpu_memory
        memory = measure_peak_gpu_memory(device)
        assert isinstance(memory, float)

    def test_condition_truth_table_false_true(self):
        """
        测试用例ID: TC-CC-006-1c
        真值表行: False, True
        """
        model = nn.Sequential()  # total > 0: False
        device = torch.device("cpu")  # device != "cuda": True

        total, trainable, ratio = count_trainable_params(model)
        assert total == 0 and ratio == 0.0

        from eval.metrics_engineering import measure_peak_gpu_memory
        memory = measure_peak_gpu_memory(device)
        assert memory == 0.0
