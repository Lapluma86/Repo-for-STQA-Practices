"""
第一阶段优化集成脚本

将以下优化集成到现有训练流程：
1. 不确定性加权多任务学习
2. 辅助任务（情绪维度、情感分类、AU预测）
3. 类别平衡损失函数（已在 heads.py 中实现）

使用方式：
    python train.py --task a1 --config configs/phase1_optimization.yaml
"""
from __future__ import annotations

import torch
import torch.nn as nn
from pathlib import Path

from common.models.grouped_model import GroupedModel
from common.models.mtl_uncertainty import (
    UncertaintyWeightedLoss,
    MultiTaskHead,
    compute_auxiliary_losses,
)


class OptimizedGroupedModel(nn.Module):
    """优化版分组模型：集成不确定性加权MTL和辅助任务 (仅 emotion_dims 弱正则化)"""
    def __init__(
        self,
        grouped_model: GroupedModel,
        participant_head: nn.Module,
        session_head: nn.Module,
        d_shared: int,
        aux_dim: int = 0,
        use_uncertainty_weighting: bool = True,
        enable_auxiliary_tasks: bool = True,
        enable_emotion_dims: bool = True,
        uw_log_var_clamp: float | None = None,
    ):
        super().__init__()
        self.grouped_model = grouped_model
        self.participant_head = participant_head
        self.session_head = session_head
        self.d_shared = d_shared
        self.aux_dim = aux_dim

        self.enable_auxiliary_tasks = enable_auxiliary_tasks
        if enable_auxiliary_tasks:
            self.aux_task_head = MultiTaskHead(
                d_in=d_shared + aux_dim,
                task_type="a1",
                enable_emotion_dims=enable_emotion_dims,
            )

        self.use_uncertainty_weighting = use_uncertainty_weighting
        if use_uncertainty_weighting:
            n_tasks = 3  # 主任务、会话任务、会话类型
            if enable_auxiliary_tasks and enable_emotion_dims:
                n_tasks += 1
            self.uncertainty_loss = UncertaintyWeightedLoss(n_tasks=n_tasks, log_var_clamp=uw_log_var_clamp)

    def forward(
        self,
        flat_batch: dict,
        n_participants: int,
        session_valid: torch.Tensor,
        aux_attrs: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        前向传播

        返回:
            outputs: 包含所有任务的预测输出
        """
        # 骨干网络输出
        backbone_out = self.grouped_model(flat_batch, n_participants, session_valid, aux_attrs)

        # 主任务和会话任务预测
        participant_logits = self.participant_head(backbone_out["participant_repr"])
        session_logits = self.session_head(backbone_out["session_reprs"])

        outputs = {
            "participant_logits": participant_logits,
            "session_logits": session_logits,
            "session_type_logits": backbone_out["session_type_logits"],
            "session_reprs": backbone_out["session_reprs"],
            "aux_logits": backbone_out.get("aux_logits"),
        }

        # 辅助任务预测
        if self.enable_auxiliary_tasks:
            aux_outputs = self.aux_task_head(backbone_out["participant_repr"])
            outputs.update(aux_outputs)

        return outputs


def compute_optimized_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    model: OptimizedGroupedModel,
    task: str,
    session_valid: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
    use_combined_loss: bool = False,
    gamma_neg: float = 2.0,
    gamma_pos: float = 0.0,
    clip: float = 0.05,
    soft_f1_weight: float = 0.3,
    use_corn_loss: bool = False,
    use_qwk_aux: bool = False,
    qwk_weight: float = 0.3,
    session_loss_weight: float = 0.5,
    session_type_loss_weight: float = 0.15,
    emotion_dims_weight: float = 0.05,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    计算优化后的总损失

    返回:
        total_loss: 总损失
        loss_dict: 各任务损失的详细信息（用于日志）
    """
    from common.models.heads import a1_loss, a2_ordinal_loss
    import torch.nn.functional as F

    losses = []
    loss_dict = {}

    # 1. 主任务损失（参与者级）
    loss_components: dict[str, float] = {}
    if task == "a1":
        main_loss = a1_loss(
            outputs["participant_logits"],
            targets["participant_y"],
            pos_weight=pos_weight,
            label_smoothing=label_smoothing,
            use_combined=use_combined_loss,
            gamma_neg=gamma_neg,
            gamma_pos=gamma_pos,
            clip=clip,
            soft_f1_weight=soft_f1_weight,
        )
    else:
        main_loss = a2_ordinal_loss(
            outputs["participant_logits"],
            targets["participant_y"],
            pos_weight=pos_weight,
            label_smoothing=label_smoothing,
            use_corn=use_corn_loss,
            use_qwk=use_qwk_aux,
            qwk_weight=qwk_weight,
            loss_components=loss_components,
        )
    losses.append(main_loss)
    loss_dict["main_loss"] = main_loss.item()
    loss_dict.update({f"main_{k}": v for k, v in loss_components.items()})

    # 2. 会话级任务损失
    valid_session_mask = session_valid.reshape(-1).bool()
    has_valid_sessions = bool(valid_session_mask.any().item())

    if has_valid_sessions:
        s_logits = outputs["session_logits"][valid_session_mask]
        if task == "a1":
            s_targets = targets["participant_y"].unsqueeze(1).expand(-1, 4, -1).reshape(-1, 3)[valid_session_mask]
            sess_loss = a1_loss(
                s_logits, s_targets,
                pos_weight=pos_weight,
                label_smoothing=label_smoothing,
                use_combined=use_combined_loss,
                gamma_neg=gamma_neg,
                gamma_pos=gamma_pos,
                clip=clip,
                soft_f1_weight=soft_f1_weight,
            )
        else:
            s_targets = targets["participant_y"].unsqueeze(1).expand(-1, 4, -1).reshape(-1, 21)[valid_session_mask]
            sess_lc: dict[str, float] = {}
            sess_loss = a2_ordinal_loss(
                s_logits, s_targets,
                pos_weight=pos_weight,
                label_smoothing=label_smoothing,
                use_corn=use_corn_loss,
                use_qwk=use_qwk_aux,
                qwk_weight=qwk_weight,
                loss_components=sess_lc,
            )

        # 会话类型分类损失
        type_loss = F.cross_entropy(
            outputs["session_type_logits"][valid_session_mask],
            targets["session_types"][valid_session_mask],
        )
    else:
        sess_loss = main_loss.new_zeros(())
        type_loss = main_loss.new_zeros(())

    losses.append(sess_loss)
    losses.append(type_loss)
    loss_dict["session_loss"] = sess_loss.item()
    loss_dict["session_type_loss"] = type_loss.item()
    loss_dict.update({f"sess_{k}": v for k, v in sess_lc.items()})

    # 3. 辅助任务损失 (仅 emotion_dims 弱正则化)
    if model.enable_auxiliary_tasks:
        aux_targets = targets.get("auxiliary_targets")
        aux_outputs = {k: v for k, v in outputs.items() if k == "emotion_dims"}
        aux_losses = compute_auxiliary_losses(aux_outputs, aux_targets)
        for key, loss in aux_losses.items():
            losses.append(loss)
            loss_dict[f"aux_{key}"] = loss.item()

    # 4. 计算总损失
    if model.use_uncertainty_weighting:
        total_loss, weights = model.uncertainty_loss(losses)
        loss_dict.update(weights)
    else:
        total_loss = losses[0]
        total_loss = total_loss + session_loss_weight * losses[1]
        total_loss = total_loss + session_type_loss_weight * losses[2]

        if model.enable_auxiliary_tasks:
            for i in range(3, len(losses)):
                total_loss = total_loss + emotion_dims_weight * losses[i]

    loss_dict["total_loss"] = total_loss.item()
    return total_loss, loss_dict


def create_optimized_model(
    grouped_model: GroupedModel,
    participant_head: nn.Module,
    session_head: nn.Module,
    cfg: dict,
    d_shared: int,
    aux_dim: int = 0,
) -> OptimizedGroupedModel:
    """
    创建优化版模型

    参数:
        grouped_model: 原始分组模型
        participant_head: 参与者级任务头
        session_head: 会话级任务头
        cfg: 配置字典
        d_shared: 共享表示维度
        aux_dim: 辅助属性维度

    返回:
        优化版模型
    """
    return OptimizedGroupedModel(
        grouped_model=grouped_model,
        participant_head=participant_head,
        session_head=session_head,
        d_shared=d_shared,
        aux_dim=aux_dim,
        use_uncertainty_weighting=cfg.get("use_uncertainty_weighting", True),
        enable_auxiliary_tasks=cfg.get("enable_auxiliary_tasks", True),
        enable_emotion_dims=cfg.get("enable_emotion_dims", True),
        uw_log_var_clamp=cfg.get("uw_log_var_clamp", None),
    )


# ============================================================
# 使用示例
# ============================================================
"""
在 runner.py 中集成：

from common.models.phase1_integration import (
    create_optimized_model,
    compute_optimized_loss,
)

# 创建模型
optimized_model = create_optimized_model(
    grouped_model=grouped_model,
    participant_head=participant_head,
    session_head=session_head,
    cfg=cfg,
    d_shared=bb_cfg.d_shared,
    aux_dim=aux_dim,
).to(device)

# 训练循环中
outputs = optimized_model(flat_batch, B, session_valid, aux_attrs)
loss, loss_dict = compute_optimized_loss(
    outputs=outputs,
    targets={
        "participant_y": targets,
        "session_types": session_types,
        "auxiliary_targets": aux_targets,  # 可选
    },
    model=optimized_model,
    task=task,
    session_valid=session_valid,
    pos_weight=pos_weight_t,
    label_smoothing=cfg.get("label_smoothing", 0.0),
    use_combined_loss=cfg.get("use_combined_loss", False),
    gamma_neg=cfg.get("gamma_neg", 2.0),
    gamma_pos=cfg.get("gamma_pos", 0.0),
    clip=cfg.get("clip", 0.05),
    soft_f1_weight=cfg.get("soft_f1_weight", 0.3),
    use_corn_loss=cfg.get("use_corn_loss", False),
    use_qwk_aux=cfg.get("use_qwk_aux", False),
    qwk_weight=cfg.get("qwk_weight", 0.3),
)
"""
