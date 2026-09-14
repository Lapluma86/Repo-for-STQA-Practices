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
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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
@pytest.mark.phase1
class TestStatementCoverage:
    """语句覆盖测试类"""

    # ===== 测试用例 TC-SC-001: loss_l2 语句覆盖 =====

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

    def test_loss_l2_empty_features(self):
        """
        测试用例ID: TC-SC-001-2
        测试目的: 覆盖空特征列表的情况
        """
        feature_s = []
        feature_t = []

        loss = loss_l2(feature_s, feature_t)

        # 空列表应该返回 0
        assert loss == 0.0

    def test_loss_l2_single_scale(self):
        """
        测试用例ID: TC-SC-001-3
        测试目的: 测试单尺度特征，覆盖循环一次的情况
        """
        feature_s = [torch.randn(2, 64, 32, 32)]
        feature_t = [torch.randn(2, 64, 32, 32)]

        loss = loss_l2(feature_s, feature_t)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    # ===== 测试用例 TC-SC-002: loss_distil 语句覆盖 =====

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

    def test_loss_distil_identical_features(self):
        """
        测试用例ID: TC-SC-002-2
        测试目的: 测试相同特征，余弦损失应该接近0
        """
        feature = [torch.randn(2, 64, 32, 32)]

        loss = loss_distil(feature, feature)

        # 相同特征的余弦损失应该非常小
        assert loss.item() < 0.01

    # ===== 测试用例 TC-SC-003: loss_distil_p 语句覆盖 =====

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

    # ===== 测试用例 TC-SC-004: loss_distil_pixel 语句覆盖 =====

    def test_loss_distil_pixel_basic(self):
        """
        测试用例ID: TC-SC-004-1
        测试目的: 覆盖 loss_distil_pixel 的所有语句
        覆盖语句:
          - line 82-93: 初始化和池化层
          - line 95: for i in range(len(feature_s))
          - line 97-98: f_s = feature_s[i], f_t = feature_t[i]
          - line 102-104: 池化操作（i == 0 or i == 1）
          - line 107: 计算相似度矩阵
          - line 110: loss += loss_type(cos_sim_s, cos_sim_t)
          - line 112: return loss
        """
        # 使用较小的特征图以节省计算
        feature_s = [
            torch.randn(1, 64, 16, 16),  # 会被池化 4x
            torch.randn(1, 128, 8, 8),   # 会被池化 2x
            torch.randn(1, 256, 4, 4)    # 不池化
        ]
        feature_t = [
            torch.randn(1, 64, 16, 16),
            torch.randn(1, 128, 8, 8),
            torch.randn(1, 256, 4, 4)
        ]

        loss = loss_distil_pixel(feature_s, feature_t)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_loss_distil_pixel_branch_coverage(self):
        """
        测试用例ID: TC-SC-004-2
        测试目的: 覆盖池化分支（i==0, i==1）和非池化分支（i==2）
        """
        # 三个尺度分别覆盖不同的分支
        feature_s = [
            torch.randn(1, 64, 16, 16),   # i=0: 执行 pool_s[0]
            torch.randn(1, 128, 8, 8),    # i=1: 执行 pool_s[1]
            torch.randn(1, 256, 4, 4)     # i=2: 不执行池化
        ]
        feature_t = [
            torch.randn(1, 64, 16, 16),
            torch.randn(1, 128, 8, 8),
            torch.randn(1, 256, 4, 4)
        ]

        loss = loss_distil_pixel(feature_s, feature_t)

        assert isinstance(loss, torch.Tensor)

    # ===== 测试用例 TC-SC-005: calculate_pixel_similarity 语句覆盖 =====

    def test_calculate_pixel_similarity_basic(self):
        """
        测试用例ID: TC-SC-005-1
        测试目的: 覆盖 calculate_pixel_similarity 的所有语句
        覆盖语句:
          - line 124-126: 学生特征处理
          - line 127: 创建空张量
          - line 129-134: 学生相似度计算循环
          - line 137-147: 教师相似度计算
          - line 148: return
        """
        f_s = torch.randn(2, 64, 4, 4)
        f_t = torch.randn(2, 64, 4, 4)

        cos_sim_s, cos_sim_t = calculate_pixel_similarity(f_s, f_t)

        # 验证输出形状
        assert cos_sim_s.shape == (2, 16, 16)  # 4*4=16 pixels
        assert cos_sim_t.shape == (2, 16, 16)

        # 验证余弦相似度范围 [-1, 1]
        assert cos_sim_s.min() >= -1.01  # 允许小误差
        assert cos_sim_s.max() <= 1.01

    def test_calculate_pixel_similarity_batch_loop(self):
        """
        测试用例ID: TC-SC-005-2
        测试目的: 覆盖批次循环（line 129, 141）
        """
        # 测试多个批次
        f_s = torch.randn(3, 32, 4, 4)
        f_t = torch.randn(3, 32, 4, 4)

        cos_sim_s, cos_sim_t = calculate_pixel_similarity(f_s, f_t)

        assert cos_sim_s.shape[0] == 3  # batch size
        assert cos_sim_t.shape[0] == 3

    # ===== 测试用例 TC-SC-006: calculate_pixel_similarity_f 语句覆盖 =====

    def test_calculate_pixel_similarity_f_basic(self):
        """
        测试用例ID: TC-SC-006-1
        测试目的: 覆盖 calculate_pixel_similarity_f 的所有语句
        """
        f = torch.randn(2, 64, 4, 4)

        cos_sim = calculate_pixel_similarity_f(f)

        assert cos_sim.shape == (2, 16, 16)
        assert cos_sim.min() >= -1.01
        assert cos_sim.max() <= 1.01

    # ===== 测试用例 TC-SC-007: calculate_pixel_similarity_f_2 语句覆盖 =====

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

    # ===== 测试用例 TC-SC-008: count_trainable_params 语句覆盖 =====

    def test_count_trainable_params_basic(self):
        """
        测试用例ID: TC-SC-008-1
        测试目的: 覆盖 count_trainable_params 的所有语句
        覆盖语句:
          - line 17: total = sum(...)
          - line 18: trainable = sum(...)
          - line 19: ratio = trainable / total if total > 0 else 0.0
          - line 20: return
        """
        # 创建简单模型
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )

        total, trainable, ratio = count_trainable_params(model)

        assert total > 0
        assert trainable > 0
        assert 0 <= ratio <= 1
        assert total == trainable  # 默认所有参数可训练

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

    def test_count_trainable_params_empty_model(self):
        """
        测试用例ID: TC-SC-008-3
        测试目的: 测试空模型，覆盖 total == 0 的分支
        """
        model = nn.Sequential()

        total, trainable, ratio = count_trainable_params(model)

        assert total == 0
        assert trainable == 0
        assert ratio == 0.0  # 覆盖 else 分支

    # ===== 测试用例 TC-SC-009: measure_inference_latency 语句覆盖 =====

    def test_measure_inference_latency_basic(self):
        """
        测试用例ID: TC-SC-009-1
        测试目的: 覆盖 measure_inference_latency 的所有语句
        覆盖语句:
          - line 31: assert
          - line 32: model.eval()
          - line 33: use_cuda = device.type == "cuda"
          - line 34-36: warmup loop
          - line 37-38: CUDA sync (if use_cuda)
          - line 39-42: timing loop
          - line 43-44: CUDA sync
          - line 45: return
        """
        model = nn.Linear(10, 5)
        sample_input = (torch.randn(1, 10),)
        device = torch.device("cpu")

        latency = measure_inference_latency(
            model, sample_input, device,
            n_warmup=5, n_run=10
        )

        assert isinstance(latency, float)
        assert latency > 0  # 应该有正的时延

    def test_measure_inference_latency_cuda_branch(self):
        """
        测试用例ID: TC-SC-009-2
        测试目的: 覆盖 CUDA 分支（如果有 GPU）
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        model = nn.Linear(10, 5).cuda()
        sample_input = (torch.randn(1, 10).cuda(),)
        device = torch.device("cuda")

        latency = measure_inference_latency(
            model, sample_input, device,
            n_warmup=5, n_run=10
        )

        assert latency > 0

    def test_measure_inference_latency_assertion(self):
        """
        测试用例ID: TC-SC-009-3
        测试目的: 覆盖 assertion 失败的情况
        """
        model = nn.Linear(10, 5)
        sample_input = torch.randn(1, 10)  # 不是 tuple/list
        device = torch.device("cpu")

        with pytest.raises(AssertionError):
            measure_inference_latency(model, sample_input, device)

    # ===== 测试用例 TC-SC-010: measure_peak_gpu_memory 语句覆盖 =====

    def test_measure_peak_gpu_memory_cpu(self):
        """
        测试用例ID: TC-SC-010-1
        测试目的: 覆盖 CPU 设备分支
        覆盖语句:
          - line 50-51: if device.type != "cuda": return 0.0
        """
        device = torch.device("cpu")

        memory = measure_peak_gpu_memory(device)

        assert memory == 0.0

    def test_measure_peak_gpu_memory_cuda(self):
        """
        测试用例ID: TC-SC-010-2
        测试目的: 覆盖 CUDA 设备分支
        覆盖语句:
          - line 52: return torch.cuda.max_memory_allocated(device) / 1024 ** 2
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = torch.device("cuda")
        torch.cuda.reset_peak_memory_stats(device)

        # 分配一些内存
        _ = torch.randn(1000, 1000).cuda()

        memory = measure_peak_gpu_memory(device)

        assert memory > 0  # 应该有内存使用

    # ===== 测试用例 TC-SC-011: compute_fp_per_image 语句覆盖 =====

    def test_compute_fp_per_image_basic(self):
        """
        测试用例ID: TC-SC-011-1
        测试目的: 覆盖 compute_fp_per_image 的所有语句
        覆盖语句:
          - line 59-60: 转换为 numpy 数组
          - line 61: normal_mask = labels == 0
          - line 62: n_normal = int(normal_mask.sum())
          - line 63-64: if n_normal == 0: return 0.0
          - line 65: return float(...)
        """
        scores = np.array([0.1, 0.3, 0.7, 0.9, 0.2])
        labels = np.array([0, 0, 1, 1, 0])  # 0=正常, 1=异常
        threshold = 0.5

        fp_rate = compute_fp_per_image(scores, labels, threshold)

        # 正常样本: [0.1, 0.3, 0.2], 大于0.5的: 0个
        assert fp_rate == 0.0

    def test_compute_fp_per_image_with_false_positives(self):
        """
        测试用例ID: TC-SC-011-2
        测试目的: 测试有误报的情况
        """
        scores = np.array([0.1, 0.6, 0.7, 0.9, 0.8])
        labels = np.array([0, 0, 1, 1, 0])
        threshold = 0.5

        fp_rate = compute_fp_per_image(scores, labels, threshold)

        # 正常样本: [0.1, 0.6, 0.8], 大于0.5的: 2个
        # FP率 = 2/3 ≈ 0.667
        assert abs(fp_rate - 2/3) < 0.01

    def test_compute_fp_per_image_no_normal_samples(self):
        """
        测试用例ID: TC-SC-011-3
        测试目的: 覆盖没有正常样本的分支
        """
        scores = np.array([0.7, 0.8, 0.9])
        labels = np.array([1, 1, 1])  # 全是异常
        threshold = 0.5

        fp_rate = compute_fp_per_image(scores, labels, threshold)

        # 没有正常样本，返回 0.0
        assert fp_rate == 0.0

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
