# -*- coding: utf-8 -*-
"""
白盒测试 - 路径覆盖测试
测试目标：覆盖程序中从入口到出口的不同执行路径

测试策略：
1. 识别独立路径（基于控制流图的圈复杂度）
2. 设计测试用例覆盖每条独立路径
3. 对于复杂函数，至少覆盖主要路径

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

from utils.losses import loss_l2, loss_distil, loss_distil_pixel
from eval.metrics_engineering import (
    count_trainable_params,
    measure_inference_latency,
    measure_peak_gpu_memory,
    compute_fp_per_image
)


@pytest.mark.whitebox
@pytest.mark.phase1
class TestPathCoverage:
    """路径覆盖测试类"""

    # ===== 测试用例 TC-PC-001: count_trainable_params 路径覆盖 =====
    # 路径1: total > 0 → 执行除法 → return
    # 路径2: total == 0 → 返回0.0 → return

    def test_count_trainable_params_path_1(self):
        """
        测试用例ID: TC-PC-001-1
        路径: 入口 → total>0(True) → ratio=trainable/total → return
        测试目的: 覆盖正常计算路径
        """
        model = nn.Linear(10, 5)

        total, trainable, ratio = count_trainable_params(model)

        # 路径验证
        assert total > 0
        assert trainable > 0
        assert ratio == trainable / total

    def test_count_trainable_params_path_2(self):
        """
        测试用例ID: TC-PC-001-2
        路径: 入口 → total>0(False) → ratio=0.0 → return
        测试目的: 覆盖空模型路径
        """
        model = nn.Sequential()

        total, trainable, ratio = count_trainable_params(model)

        # 路径验证
        assert total == 0
        assert trainable == 0
        assert ratio == 0.0

    # ===== 测试用例 TC-PC-002: measure_peak_gpu_memory 路径覆盖 =====
    # 路径1: device.type != "cuda" → return 0.0
    # 路径2: device.type == "cuda" → 计算显存 → return

    def test_measure_peak_gpu_memory_path_1(self):
        """
        测试用例ID: TC-PC-002-1
        路径: 入口 → device!="cuda"(True) → return 0.0
        测试目的: CPU设备路径
        """
        device = torch.device("cpu")

        memory = measure_peak_gpu_memory(device)

        # 路径验证
        assert memory == 0.0

    def test_measure_peak_gpu_memory_path_2(self):
        """
        测试用例ID: TC-PC-002-2
        路径: 入口 → device!="cuda"(False) → 计算max_memory → return
        测试目的: CUDA设备路径
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = torch.device("cuda")
        torch.cuda.reset_peak_memory_stats(device)
        _ = torch.randn(100, 100).cuda()

        memory = measure_peak_gpu_memory(device)

        # 路径验证
        assert memory > 0

    # ===== 测试用例 TC-PC-003: compute_fp_per_image 路径覆盖 =====
    # 路径1: n_normal == 0 → return 0.0
    # 路径2: n_normal > 0 → 计算误报率 → return

    def test_compute_fp_per_image_path_1(self):
        """
        测试用例ID: TC-PC-003-1
        路径: 入口 → n_normal==0(True) → return 0.0
        测试目的: 无正常样本路径
        """
        scores = np.array([0.8, 0.9])
        labels = np.array([1, 1])
        threshold = 0.5

        fp_rate = compute_fp_per_image(scores, labels, threshold)

        # 路径验证
        assert fp_rate == 0.0

    def test_compute_fp_per_image_path_2(self):
        """
        测试用例ID: TC-PC-003-2
        路径: 入口 → n_normal==0(False) → 计算FP → return
        测试目的: 有正常样本路径
        """
        scores = np.array([0.6, 0.3, 0.8])
        labels = np.array([0, 0, 1])
        threshold = 0.5

        fp_rate = compute_fp_per_image(scores, labels, threshold)

        # 路径验证: 正常样本[0.6, 0.3], >0.5的1个, FP=1/2=0.5
        assert abs(fp_rate - 0.5) < 0.01

    # ===== 测试用例 TC-PC-004: measure_inference_latency 路径覆盖 =====
    # 路径1: 入口 → assert失败 → 抛出异常
    # 路径2: 入口 → assert通过 → use_cuda=False → 不同步 → return
    # 路径3: 入口 → assert通过 → use_cuda=True → 同步 → return

    def test_measure_inference_latency_path_1(self):
        """
        测试用例ID: TC-PC-004-1
        路径: 入口 → assert失败 → AssertionError
        测试目的: 断言失败路径
        """
        model = nn.Linear(10, 5)
        sample_input = torch.randn(1, 10)  # 不是tuple/list
        device = torch.device("cpu")

        with pytest.raises(AssertionError):
            measure_inference_latency(model, sample_input, device)

    def test_measure_inference_latency_path_2(self):
        """
        测试用例ID: TC-PC-004-2
        路径: 入口 → assert通过 → use_cuda=False → warmup → timing(no sync) → return
        测试目的: CPU路径
        """
        model = nn.Linear(10, 5)
        sample_input = (torch.randn(1, 10),)
        device = torch.device("cpu")

        latency = measure_inference_latency(
            model, sample_input, device,
            n_warmup=2, n_run=5
        )

        # 路径验证
        assert latency > 0

    def test_measure_inference_latency_path_3(self):
        """
        测试用例ID: TC-PC-004-3
        路径: 入口 → assert通过 → use_cuda=True → warmup → sync → timing → sync → return
        测试目的: CUDA路径（包含同步）
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

        # 路径验证
        assert latency > 0

    # ===== 测试用例 TC-PC-005: loss_distil_pixel 路径覆盖 =====
    # 路径1: 空列表 → return 0
    # 路径2: i=0 → 池化 → 计算相似度 → return
    # 路径3: i=1 → 池化 → 计算相似度 → return
    # 路径4: i=2 → 不池化 → 计算相似度 → return
    # 路径5: 多尺度混合 → 多次迭代 → return

    def test_loss_distil_pixel_path_1(self):
        """
        测试用例ID: TC-PC-005-1
        路径: 入口 → len(feature_s)=0 → 循环不执行 → return loss=0
        测试目的: 空列表路径
        """
        feature_s = []
        feature_t = []

        loss = loss_distil_pixel(feature_s, feature_t)

        # 路径验证
        assert loss == 0.0

    def test_loss_distil_pixel_path_2(self):
        """
        测试用例ID: TC-PC-005-2
        路径: 入口 → i=0 → pool_s[0] → 计算相似度 → loss累加 → return
        测试目的: 仅第0尺度路径
        """
        feature_s = [torch.randn(1, 64, 16, 16)]
        feature_t = [torch.randn(1, 64, 16, 16)]

        loss = loss_distil_pixel(feature_s, feature_t)

        # 路径验证
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_loss_distil_pixel_path_3(self):
        """
        测试用例ID: TC-PC-005-3
        路径: 入口 → i=0(池化) → i=1(池化) → 两次累加 → return
        测试目的: 0和1尺度路径
        """
        feature_s = [
            torch.randn(1, 64, 16, 16),
            torch.randn(1, 128, 8, 8)
        ]
        feature_t = [
            torch.randn(1, 64, 16, 16),
            torch.randn(1, 128, 8, 8)
        ]

        loss = loss_distil_pixel(feature_s, feature_t)

        # 路径验证
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_loss_distil_pixel_path_4(self):
        """
        测试用例ID: TC-PC-005-4
        路径: 入口 → i=2 → 不池化 → 计算相似度 → return
        测试目的: 第2尺度不池化路径
        """
        feature_s = [
            torch.randn(1, 64, 16, 16),
            torch.randn(1, 128, 8, 8),
            torch.randn(1, 256, 4, 4)
        ]
        feature_t = [
            torch.randn(1, 64, 16, 16),
            torch.randn(1, 128, 8, 8),
            torch.randn(1, 256, 4, 4)
        ]

        loss = loss_distil_pixel(feature_s, feature_t)

        # 路径验证：所有三个尺度都被处理
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_loss_distil_pixel_path_5(self):
        """
        测试用例ID: TC-PC-005-5
        路径: 入口 → i=0(池化) → i=1(池化) → i=2(不池化) → 三次累加 → return
        测试目的: 完整多尺度路径
        """
        feature_s = [
            torch.randn(1, 64, 16, 16),   # i=0: 池化4x
            torch.randn(1, 128, 8, 8),    # i=1: 池化2x
            torch.randn(1, 256, 4, 4)     # i=2: 不池化
        ]
        feature_t = [
            torch.randn(1, 64, 16, 16),
            torch.randn(1, 128, 8, 8),
            torch.randn(1, 256, 4, 4)
        ]

        loss = loss_distil_pixel(feature_s, feature_t)

        # 路径验证
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    # ===== 测试用例 TC-PC-006: loss_l2 循环路径覆盖 =====
    # 路径1: 0次循环
    # 路径2: 1次循环
    # 路径3: n次循环

    def test_loss_l2_path_zero_iterations(self):
        """
        测试用例ID: TC-PC-006-1
        路径: 入口 → len=0 → 循环0次 → return 0.0
        """
        feature_s = []
        feature_t = []

        loss = loss_l2(feature_s, feature_t)

        assert loss == 0.0

    def test_loss_l2_path_one_iteration(self):
        """
        测试用例ID: TC-PC-006-2
        路径: 入口 → len=1 → 循环1次 → 累加 → return
        """
        feature_s = [torch.randn(2, 64, 32, 32)]
        feature_t = [torch.randn(2, 64, 32, 32)]

        loss = loss_l2(feature_s, feature_t)

        assert isinstance(loss, torch.Tensor)

    def test_loss_l2_path_multiple_iterations(self):
        """
        测试用例ID: TC-PC-006-3
        路径: 入口 → len=3 → 循环3次 → 多次累加 → return
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

        loss = loss_l2(feature_s, feature_t)

        assert isinstance(loss, torch.Tensor)

    # ===== 测试用例 TC-PC-007: 组合路径测试 =====

    def test_combined_path_all_cpu(self):
        """
        测试用例ID: TC-PC-007-1
        路径组合: 所有CPU路径
        测试目的: 验证纯CPU执行路径的组合
        """
        # count_trainable_params: 路径2 (total=0)
        model_empty = nn.Sequential()
        total, trainable, ratio = count_trainable_params(model_empty)
        assert total == 0

        # measure_peak_gpu_memory: 路径1 (CPU)
        device = torch.device("cpu")
        memory = measure_peak_gpu_memory(device)
        assert memory == 0.0

        # compute_fp_per_image: 路径1 (n_normal=0)
        scores = np.array([0.8])
        labels = np.array([1])
        fp_rate = compute_fp_per_image(scores, labels, 0.5)
        assert fp_rate == 0.0

    def test_combined_path_all_cuda(self):
        """
        测试用例ID: TC-PC-007-2
        路径组合: 所有CUDA路径
        测试目的: 验证纯CUDA执行路径的组合
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # count_trainable_params: 路径1 (total>0)
        model = nn.Linear(10, 5).cuda()
        total, trainable, ratio = count_trainable_params(model)
        assert total > 0

        # measure_inference_latency: 路径3 (CUDA)
        sample_input = (torch.randn(1, 10).cuda(),)
        device = torch.device("cuda")
        latency = measure_inference_latency(
            model, sample_input, device,
            n_warmup=2, n_run=5
        )
        assert latency > 0

        # measure_peak_gpu_memory: 路径2 (CUDA)
        torch.cuda.reset_peak_memory_stats(device)
        _ = torch.randn(100, 100).cuda()
        memory = measure_peak_gpu_memory(device)
        assert memory > 0

    def test_combined_path_mixed(self):
        """
        测试用例ID: TC-PC-007-3
        路径组合: 混合路径
        测试目的: 验证不同函数的不同路径组合
        """
        # count_trainable_params: 路径1 (total>0)
        model = nn.Linear(10, 5)
        total, trainable, ratio = count_trainable_params(model)
        assert total > 0

        # measure_inference_latency: 路径2 (CPU)
        sample_input = (torch.randn(1, 10),)
        device = torch.device("cpu")
        latency = measure_inference_latency(
            model, sample_input, device,
            n_warmup=2, n_run=5
        )
        assert latency > 0

        # compute_fp_per_image: 路径2 (n_normal>0)
        scores = np.array([0.6, 0.3])
        labels = np.array([0, 0])
        fp_rate = compute_fp_per_image(scores, labels, 0.5)
        assert isinstance(fp_rate, float)

        # loss_l2: 路径3 (多次迭代)
        feature_s = [torch.randn(2, 64, 32, 32) for _ in range(3)]
        feature_t = [torch.randn(2, 64, 32, 32) for _ in range(3)]
        loss = loss_l2(feature_s, feature_t)
        assert isinstance(loss, torch.Tensor)

    # ===== 测试用例 TC-PC-008: 异常路径覆盖 =====

    def test_exception_path_assertion_error(self):
        """
        测试用例ID: TC-PC-008-1
        路径: 入口 → 断言失败 → AssertionError
        测试目的: 异常处理路径
        """
        model = nn.Linear(10, 5)
        sample_input = "invalid"  # 无效输入
        device = torch.device("cpu")

        with pytest.raises(AssertionError):
            measure_inference_latency(model, sample_input, device)

    def test_normal_path_vs_exception_path(self):
        """
        测试用例ID: TC-PC-008-2
        测试目的: 对比正常路径和异常路径
        """
        model = nn.Linear(10, 5)
        device = torch.device("cpu")

        # 正常路径
        sample_input_valid = (torch.randn(1, 10),)
        latency = measure_inference_latency(
            model, sample_input_valid, device,
            n_warmup=2, n_run=5
        )
        assert latency > 0

        # 异常路径
        sample_input_invalid = torch.randn(1, 10)
        with pytest.raises(AssertionError):
            measure_inference_latency(model, sample_input_invalid, device)
