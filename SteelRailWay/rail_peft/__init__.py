"""Lightweight PEFT helpers for rail depth-domain adaptation."""

from .depth_affine import DepthAffinePEFT, DepthEncoderWithPEFT
from .feature_stats import (
    compute_depth_feature_stats,
    compute_teacher_feature_means,
    expand_feature_means,
    fdm_loss,
    zeros_like_feature_list,
)

__all__ = [
    "DepthAffinePEFT",
    "DepthEncoderWithPEFT",
    "compute_depth_feature_stats",
    "compute_teacher_feature_means",
    "expand_feature_means",
    "fdm_loss",
    "zeros_like_feature_list",
]
