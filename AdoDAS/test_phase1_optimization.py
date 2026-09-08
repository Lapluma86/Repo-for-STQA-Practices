#!/usr/bin/env python3
"""
第一阶段优化模块测试脚本

测试以下功能：
1. 不确定性加权损失计算
2. 辅助任务头前向传播
3. 优化损失函数（ASL, CORN, QWK）
4. 集成模型前向传播

运行方式：
    python test_phase1_optimization.py
"""
import torch
import torch.nn as nn
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from common.models.mtl_uncertainty import (
    UncertaintyWeightedLoss,
    MultiTaskHead,
    compute_auxiliary_losses,
)
from common.models.heads import a1_loss, a2_ordinal_loss


def test_uncertainty_weighting():
    """测试不确定性加权损失"""
    print("\n" + "="*60)
    print("测试 1: 不确定性加权损失")
    print("="*60)

    n_tasks = 4
    uw_loss = UncertaintyWeightedLoss(n_tasks=n_tasks)

    # 模拟各任务损失
    losses = [
        torch.tensor(0.5),  # 主任务
        torch.tensor(0.3),  # 会话任务
        torch.tensor(0.8),  # 会话类型
        torch.tensor(0.2),  # 情绪维度
    ]

    total_loss, weights = uw_loss(losses)

    print(f"输入损失: {[f'{l.item():.3f}' for l in losses]}")
    print(f"总损失: {total_loss.item():.4f}")
    print(f"各任务权重:")
    for key, val in weights.items():
        print(f"  {key}: {val:.4f}")

    # 测试反向传播
    total_loss.backward()
    print(f"梯度计算成功: log_vars.grad = {uw_loss.log_vars.grad}")

    print("✓ 不确定性加权测试通过")


def test_auxiliary_task_heads():
    """测试辅助任务头"""
    print("\n" + "="*60)
    print("测试 2: 辅助任务头")
    print("="*60)

    batch_size = 4
    d_in = 256

    aux_head = MultiTaskHead(
        d_in=d_in,
        task_type="a1",
        enable_emotion_dims=True,
    )

    x = torch.randn(batch_size, d_in)
    outputs = aux_head(x)

    print(f"输入形状: {x.shape}")
    print(f"输出:")
    for key, val in outputs.items():
        print(f"  {key}: {val.shape}")

    # 测试损失计算
    aux_targets = {
        "emotion_dims": torch.randn(batch_size, 2),
    }

    losses = compute_auxiliary_losses(outputs, aux_targets)
    print(f"辅助任务损失:")
    for key, val in losses.items():
        print(f"  {key}: {val.item():.4f}")

    print("✓ 辅助任务头测试通过")


def test_a1_loss_functions():
    """测试 A1 损失函数"""
    print("\n" + "="*60)
    print("测试 3: A1 损失函数（ASL + Soft-F1）")
    print("="*60)

    batch_size = 8
    n_classes = 3

    logits = torch.randn(batch_size, n_classes)
    targets = torch.randint(0, 2, (batch_size, n_classes)).float()

    # 测试标准 BCE
    loss_bce = a1_loss(logits, targets, use_combined=False)
    print(f"标准 BCE 损失: {loss_bce.item():.4f}")

    # 测试 ASL + Soft-F1
    loss_combined = a1_loss(
        logits, targets,
        use_combined=True,
        gamma_neg=2.0,
        gamma_pos=0.0,
        clip=0.05,
        soft_f1_weight=0.3,
    )
    print(f"ASL + Soft-F1 损失: {loss_combined.item():.4f}")

    # 测试反向传播
    loss_combined.backward()
    print(f"梯度计算成功")

    print("✓ A1 损失函数测试通过")


def test_a2_loss_functions():
    """测试 A2 损失函数"""
    print("\n" + "="*60)
    print("测试 4: A2 损失函数（CORN + QWK）")
    print("="*60)

    batch_size = 8
    n_items = 21
    n_thresholds = 3

    logits = torch.randn(batch_size, n_items, n_thresholds)
    labels = torch.randint(0, 4, (batch_size, n_items))

    # 测试标准 Ordinal BCE
    loss_ordinal = a2_ordinal_loss(
        logits, labels,
        use_corn=False,
        use_qwk=False,
    )
    print(f"标准 Ordinal BCE 损失: {loss_ordinal.item():.4f}")

    # 测试 CORN
    loss_corn = a2_ordinal_loss(
        logits, labels,
        use_corn=True,
        use_qwk=False,
    )
    print(f"+ CORN 损失: {loss_corn.item():.4f}")

    # 测试 CORN + QWK
    loss_full = a2_ordinal_loss(
        logits, labels,
        use_corn=True,
        use_qwk=True,
        qwk_weight=0.3,
    )
    print(f"+ CORN + QWK 损失: {loss_full.item():.4f}")

    # 测试反向传播
    loss_full.backward()
    print(f"梯度计算成功")

    print("✓ A2 损失函数测试通过")


def test_integrated_forward():
    """测试集成模型前向传播"""
    print("\n" + "="*60)
    print("测试 5: 集成模型前向传播")
    print("="*60)

    from common.models.mtcn_backbone import MTCNBackbone, BackboneConfig
    from common.models.grouped_model import GroupedModel
    from common.models.heads import A1Head
    from common.models.phase1_integration import create_optimized_model

    # 创建简化的骨干网络配置
    bb_cfg = BackboneConfig(
        audio_group_dims={"mfcc": 40},
        video_group_dims={"openface": 35},
        d_adapter=32,
        d_model=64,
        d_shared=64,
        tcn_layers=2,
    )

    backbone = MTCNBackbone(bb_cfg)
    grouped_model = GroupedModel(
        backbone=backbone,
        d_shared=bb_cfg.d_shared,
        aggregator_method="mean",
    )

    participant_head = A1Head(bb_cfg.d_shared)
    session_head = A1Head(bb_cfg.d_shared)

    # 创建优化模型
    cfg = {
        "use_uncertainty_weighting": True,
        "enable_auxiliary_tasks": True,
        "enable_emotion_dims": True,
    }

    optimized_model = create_optimized_model(
        grouped_model=grouped_model,
        participant_head=participant_head,
        session_head=session_head,
        cfg=cfg,
        d_shared=bb_cfg.d_shared,
        aux_dim=0,
    )

    # 模拟输入
    batch_size = 2
    n_sessions = 4
    seq_len = 100

    flat_batch = {
        "audio_groups": {
            "mfcc": torch.randn(batch_size * n_sessions, seq_len, 40),
        },
        "video_groups": {
            "openface": torch.randn(batch_size * n_sessions, seq_len, 35),
        },
        "pad_mask": torch.zeros(batch_size * n_sessions, seq_len, dtype=torch.bool),
        "vad": torch.ones(batch_size * n_sessions, seq_len),
        "qc": torch.ones(batch_size * n_sessions, seq_len),
        "session_ids": torch.randint(0, 4, (batch_size * n_sessions,)),
    }

    session_valid = torch.ones(batch_size, n_sessions, dtype=torch.bool)

    # 前向传播
    outputs = optimized_model(flat_batch, batch_size, session_valid, aux_attrs=None)

    print(f"输入: batch_size={batch_size}, n_sessions={n_sessions}, seq_len={seq_len}")
    print(f"输出:")
    for key, val in outputs.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key}: {val.shape}")

    print("✓ 集成模型前向传播测试通过")


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("第一阶段优化模块测试")
    print("="*60)

    try:
        test_uncertainty_weighting()
        test_auxiliary_task_heads()
        test_a1_loss_functions()
        test_a2_loss_functions()
        test_integrated_forward()

        print("\n" + "="*60)
        print("✓ 所有测试通过！")
        print("="*60)
        print("\n可以开始使用优化模块进行训练：")
        print("  python train.py --task a2 --config tasks/a2/mtl_full.yaml")

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
