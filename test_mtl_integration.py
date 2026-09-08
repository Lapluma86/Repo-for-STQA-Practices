#!/usr/bin/env python3
"""
测试MTL集成的完整性

验证：
1. 辅助任务标签加载
2. 数据批处理
3. 模型前向传播
4. 损失计算
"""
import sys
import torch
import numpy as np
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from common.data.grouped_dataset import GroupedParticipantDataset, grouped_collate_fn
from common.data.dataset import FeatureConfig
from common.models.mtcn_backbone import BackboneConfig, MTCNBackbone
from common.models.grouped_model import GroupedModel
from common.models.aux_encoder import AuxiliaryAttributeEncoder
from common.models.heads import A1Head, A2OrdinalHead
from common.models.phase1_integration import OptimizedGroupedModel, compute_optimized_loss


def test_auxiliary_labels_loading():
    """测试辅助任务标签加载"""
    print("=" * 60)
    print("测试1: 辅助任务标签加载")
    print("=" * 60)

    try:
        # 创建数据集
        feat_cfg = FeatureConfig()
        ds = GroupedParticipantDataset(
            'manifests/train.csv',
            feat_cfg,
            split='train',
        )

        # 获取一个样本
        sample = ds[0]

        print(f"✓ 数据集加载成功，共 {len(ds)} 个参与者")

        # 检查辅助任务标签
        if "auxiliary_targets" in sample:
            aux_targets = sample["auxiliary_targets"]
            print(f"✓ 辅助任务标签存在")
            print(f"  - emotion_dims shape: {aux_targets['emotion_dims'].shape}")
            print(f"  - emotion_dims values: {aux_targets['emotion_dims']}")
        else:
            print("✗ 辅助任务标签不存在")
            return False

        print()
        return True

    except Exception as e:
        print(f"✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_collation():
    """测试批处理"""
    print("=" * 60)
    print("测试2: 批处理")
    print("=" * 60)

    try:
        feat_cfg = FeatureConfig()
        ds = GroupedParticipantDataset(
            'manifests/train.csv',
            feat_cfg,
            split='train',
        )

        # 创建小批次
        batch_samples = [ds[i] for i in range(4)]
        batch = grouped_collate_fn(batch_samples)

        print(f"✓ 批处理成功")
        print(f"  - n_participants: {batch['n_participants']}")
        print(f"  - participant_y_a1 shape: {batch['participant_y_a1'].shape}")

        if "auxiliary_targets" in batch:
            aux_targets = batch["auxiliary_targets"]
            print(f"✓ 辅助任务标签批处理成功")
            print(f"  - emotion_dims shape: {aux_targets['emotion_dims'].shape}")
        else:
            print("✗ 辅助任务标签未批处理")
            return False

        print()
        return True

    except Exception as e:
        print(f"✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_forward():
    """测试模型前向传播"""
    print("=" * 60)
    print("测试3: 模型前向传播")
    print("=" * 60)

    try:
        # 创建数据集和批次
        feat_cfg = FeatureConfig()
        ds = GroupedParticipantDataset(
            'manifests/train.csv',
            feat_cfg,
            split='train',
        )

        batch_samples = [ds[i] for i in range(2)]
        batch = grouped_collate_fn(batch_samples)

        # 创建模型
        dims = ds.feature_dims
        audio_group_dims = {n: dims[n] for n in feat_cfg.audio_sequence_features if n in dims}
        audio_pooled_group_dims = {n: dims[n] for n in feat_cfg.audio_pooled_features if n in dims}
        video_group_dims = {n: dims[n] for n in feat_cfg.video_features if n in dims}

        bb_cfg = BackboneConfig(
            audio_group_dims=audio_group_dims,
            audio_pooled_group_dims=audio_pooled_group_dims,
            video_group_dims=video_group_dims,
            d_adapter=64,
            d_model=128,
            tcn_layers=2,
            d_shared=128,
        )

        backbone = MTCNBackbone(bb_cfg)

        # 辅助属性编码器
        aux_encoder = AuxiliaryAttributeEncoder(embed_dim=8, dropout=0.1)
        aux_dim = aux_encoder.output_dim

        grouped_model = GroupedModel(
            backbone=backbone,
            d_shared=bb_cfg.d_shared,
            aggregator_method="mlp",
            dropout=0.1,
            aux_encoder=aux_encoder,
        )

        participant_head = A1Head(bb_cfg.d_shared + aux_dim)
        session_head = A1Head(bb_cfg.d_shared)

        # 创建优化模型
        optimized_model = OptimizedGroupedModel(
            grouped_model=grouped_model,
            participant_head=participant_head,
            session_head=session_head,
            d_shared=bb_cfg.d_shared,
            aux_dim=aux_dim,
            use_uncertainty_weighting=True,
            enable_auxiliary_tasks=True,
            enable_emotion_dims=True,
        )

        print(f"✓ 模型创建成功")
        print(f"  - 参数数量: {sum(p.numel() for p in optimized_model.parameters()):,}")

        # 前向传播
        device = torch.device("cpu")
        flat_batch = batch["flat_batch"]
        session_valid = batch["session_valid"]
        aux_attrs = batch["participant_aux_attrs"]
        B = batch["n_participants"]

        outputs = optimized_model(flat_batch, B, session_valid, aux_attrs)

        print(f"✓ 前向传播成功")
        print(f"  - participant_logits shape: {outputs['participant_logits'].shape}")
        print(f"  - session_logits shape: {outputs['session_logits'].shape}")

        if "emotion_dims" in outputs:
            print(f"  - emotion_dims shape: {outputs['emotion_dims'].shape}")

        print()
        return True

    except Exception as e:
        print(f"✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_loss_computation():
    """测试损失计算"""
    print("=" * 60)
    print("测试4: 损失计算")
    print("=" * 60)

    try:
        # 创建数据集和批次
        feat_cfg = FeatureConfig()
        ds = GroupedParticipantDataset(
            'manifests/train.csv',
            feat_cfg,
            split='train',
        )

        batch_samples = [ds[i] for i in range(2)]
        batch = grouped_collate_fn(batch_samples)

        # 创建模型（复用test_model_forward的代码）
        dims = ds.feature_dims
        audio_group_dims = {n: dims[n] for n in feat_cfg.audio_sequence_features if n in dims}
        audio_pooled_group_dims = {n: dims[n] for n in feat_cfg.audio_pooled_features if n in dims}
        video_group_dims = {n: dims[n] for n in feat_cfg.video_features if n in dims}

        bb_cfg = BackboneConfig(
            audio_group_dims=audio_group_dims,
            audio_pooled_group_dims=audio_pooled_group_dims,
            video_group_dims=video_group_dims,
            d_adapter=64,
            d_model=128,
            tcn_layers=2,
            d_shared=128,
        )

        backbone = MTCNBackbone(bb_cfg)
        aux_encoder = AuxiliaryAttributeEncoder(embed_dim=8, dropout=0.1)
        aux_dim = aux_encoder.output_dim

        grouped_model = GroupedModel(
            backbone=backbone,
            d_shared=bb_cfg.d_shared,
            aggregator_method="mlp",
            dropout=0.1,
            aux_encoder=aux_encoder,
        )

        participant_head = A1Head(bb_cfg.d_shared + aux_dim)
        session_head = A1Head(bb_cfg.d_shared)

        optimized_model = OptimizedGroupedModel(
            grouped_model=grouped_model,
            participant_head=participant_head,
            session_head=session_head,
            d_shared=bb_cfg.d_shared,
            aux_dim=aux_dim,
            use_uncertainty_weighting=True,
            enable_auxiliary_tasks=True,
            enable_emotion_dims=True,
        )

        # 前向传播
        flat_batch = batch["flat_batch"]
        session_valid = batch["session_valid"]
        aux_attrs = batch["participant_aux_attrs"]
        B = batch["n_participants"]

        outputs = optimized_model(flat_batch, B, session_valid, aux_attrs)

        # 准备目标
        targets = {
            "participant_y": batch["participant_y_a1"],
            "session_types": batch["session_types"],
            "auxiliary_targets": batch["auxiliary_targets"],
        }

        # 计算损失
        loss, loss_dict = compute_optimized_loss(
            outputs=outputs,
            targets=targets,
            model=optimized_model,
            task="a1",
            session_valid=session_valid,
            use_combined_loss=True,
        )

        print(f"✓ 损失计算成功")
        print(f"  - 总损失: {loss.item():.4f}")
        print(f"  - 详细损失:")
        for key, val in loss_dict.items():
            print(f"    - {key}: {val:.4f}")

        # 测试反向传播
        loss.backward()
        print(f"✓ 反向传播成功")

        print()
        return True

    except Exception as e:
        print(f"✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 60)
    print("MTL 集成测试")
    print("=" * 60 + "\n")

    results = []

    # 运行所有测试
    results.append(("辅助任务标签加载", test_auxiliary_labels_loading()))
    results.append(("批处理", test_batch_collation()))
    results.append(("模型前向传播", test_model_forward()))
    results.append(("损失计算", test_loss_computation()))

    # 总结
    print("=" * 60)
    print("测试总结")
    print("=" * 60)

    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{status}: {name}")

    all_passed = all(r[1] for r in results)

    print()
    if all_passed:
        print("🎉 所有测试通过！MTL集成完成。")
        print()
        print("下一步：运行训练")
        print("  python train.py --task a2 --config tasks/a2/mtl_full.yaml")
    else:
        print("❌ 部分测试失败，请检查错误信息。")

    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
