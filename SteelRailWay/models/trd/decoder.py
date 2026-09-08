# -*- coding: utf-8 -*-
"""
解码器（Decoder）模块定义文件。
在 Teacher-Reverse-Distillation（TRD）范式中，解码器接收编码器最深层的瓶颈特征，
通过反向（de-）ResNet 逐级上采样还原出多尺度特征，再与编码器的特征做比较，
差异越大越可能是异常区域。本文件提供两种解码器：
    1. ResNet50Decoder：单模态（仅 RGB）反向 ResNet 解码器。
    2. ResNet50DualModalDecoder：双模态（RGB + 深度/法向辅助分支）解码器，
       通过 Projector 把辅助模态映射到与 RGB 特征同维度，
       并在 BN 和解码阶段做特征融合。
"""

import torch

from .resnet import bn, bn_fuse  # bn: 单模态瓶颈 BN；bn_fuse: 双模态融合 BN
from .de_resnet import de_wide_resnet50_2, de_wide_resnet50_2_skip  # 反向 Wide-ResNet（可选 skip 融合）
from .projector import *  # 引入各种 Projector（辅助模态 -> 主模态特征空间映射）


def init_weight(m):
    """Xavier 正态初始化，统一处理 Linear / Conv2d 两类层。"""
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)


class ResNet50Decoder(torch.nn.Module):
    """单模态（RGB）反向 Wide-ResNet50_2 解码器。

    工作流程：
        1. 通过 ``bn`` 对编码器的多尺度特征做瓶颈处理（通道/尺度压缩）。
        2. 经 ``de_wide_resnet50_2`` 逐级上采样还原特征层次。
    """

    def __init__(self, pretrained=False):
        super(ResNet50Decoder, self).__init__()
        self.bn = bn()  # 瓶颈模块（OCBE, One-Class Bottleneck Embedding 等结构）
        self.decoder = de_wide_resnet50_2(pretrained=pretrained)

    def forward(self, x):
        # x 通常是编码器输出的多尺度特征列表 [feat1, feat2, feat3]
        x = self.bn(x)          # 融合 / 瓶颈
        x = self.decoder(x)     # 反向 ResNet 解码，得到与编码器对齐的重建特征
        return x


MODULE_ABLATION_MODES = ("full", "no_cf", "no_ca", "no_cf_ca")


class ResNet50DualModalDecoder(torch.nn.Module):
    """双模态解码器（RGB 主 + 深度或法向辅助）。

    核心思想：
        - 将辅助模态（深度/法向图）的特征通过两条 Projector 支路映射：
            * ``projector_filter``：滤波投影，用于瓶颈 BN 中做跨模态融合。
            * ``projector_amply``：放大/增强投影，用于解码阶段 skip 融合。
        - 解码时，瓶颈 BN 结合 filter 投影；在每一级解码块中再注入 amply 投影。

    返回：
        proj_filter : 供瓶颈阶段融合的辅助投影特征列表
        proj_amply  : 供解码阶段融合的辅助投影特征列表
        x           : 解码后的主模态重建特征（多尺度）
        x_amply     : 解码过程中融合后的中间特征（用于多任务监督）
    """

    def __init__(self, pretrained=False, module_ablation="full"):
        super(ResNet50DualModalDecoder, self).__init__()
        self.set_module_ablation(module_ablation)
        self.bn = bn_fuse()                                    # 支持跨模态融合的 BN
        self.decoder = de_wide_resnet50_2_skip(pretrained=pretrained)  # 支持 skip 融合的反向 ResNet
        # self.projector_filter = NormalProjector()  # 历史版本：等比例 Projector
        self.projector_filter = NormalProjector_8()            # 下采样 8× 后再上采样，强化瓶颈处的跨模态对齐
        self.projector_amply = Projector_Amply_upsample()      # 保持尺寸，扩大中间通道再压回，起“放大”作用

    def set_module_ablation(self, mode):
        """Set post-hoc inference ablation mode for CF/CA paths."""
        if mode not in MODULE_ABLATION_MODES:
            raise ValueError(f"unknown module_ablation: {mode}")
        self.module_ablation = mode

    def _disable_cf(self):
        return self.module_ablation in {"no_cf", "no_cf_ca"}

    def _disable_ca(self):
        return self.module_ablation in {"no_ca", "no_cf_ca"}

    def forward(self, x, x_assist, attn=False, noise=False):
        """
        参数：
            x        : 主模态（RGB）编码器多尺度特征
            x_assist : 辅助模态（depth/normal）编码器多尺度特征
            attn     : 占位参数（历史实验用于开关注意力）
            noise    : 占位参数（历史实验用于注入噪声）
        """
        x_encoder = x  # 保留引用，便于阅读（当前未使用）

        # 辅助模态两路 Projector 输出
        proj_filter, x_assist_proj = self.projector_filter(x_assist)
        proj_amply = self.projector_amply(x_assist)
        proj_filter_for_bn = (
            [torch.zeros_like(t) for t in proj_filter]
            if self._disable_cf()
            else proj_filter
        )

        # 瓶颈处融合：RGB 主特征 + 滤波投影
        x_bn = self.bn(x, proj_filter_for_bn)

        # 解码阶段：在每级逐步注入放大投影，得到重建特征 x 和融合中间特征 x_amply
        x_amply, x = self.decoder.forward_fuse(
            x_bn,
            proj_amply,
            disable_ca=self._disable_ca(),
        )

        return proj_filter_for_bn, proj_amply, x, x_amply
