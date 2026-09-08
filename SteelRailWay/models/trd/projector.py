# -*- coding: utf-8 -*-
"""
Projector（投影器）模块集合。

在双模态异常检测框架里，Projector 用于把辅助模态（如深度图、法向量图）
经过若干卷积变换后，投影到与 RGB 主分支特征相同或相容的通道数和空间分辨率，
便于后续做特征融合（filter / amply / skip 融合等）。

本文件提供三种 Projector：
    - Projector：等尺寸等通道的轻量投影（3x3 + 1x1）。
    - Projector_Amply_upsample：通道先放大再压缩的"放大"投影，用于解码阶段增强特征。
    - NormalProjector_8：通过逐步下采样 8 倍再逐级反卷积还原的大感受野投影。
所有模块都对 Conv 使用 Kaiming 初始化，对 BN 使用常数初始化（weight=1, bias=0）。
"""

import torch
import torch.nn as nn


class Projector(torch.nn.Module):
    """等尺寸投影器：保留空间分辨率和通道数，仅通过卷积学习跨模态映射。

    三条支路分别处理三种尺度的特征：
        conv_a -> 256 通道特征（浅层）
        conv_b -> 512 通道特征（中层）
        conv_c -> 1024 通道特征（深层）
    每条支路结构：3x3 Conv + BN + ReLU  再接 1x1 Conv + BN + ReLU。
    """

    def __init__(self):
        super(Projector, self).__init__()

        # 浅层支路：处理 256 通道特征图（最高分辨率）
        self.conv_a = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True)
        )
        # 中层支路：处理 512 通道特征图
        self.conv_b = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, kernel_size=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True)
        )
        # 深层支路：处理 1024 通道特征图（最低分辨率）
        self.conv_c = torch.nn.Sequential(
            torch.nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(1024, 1024, kernel_size=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU(inplace=True)
        )

        # 统一初始化：Conv 使用 Kaiming（适配 ReLU），BN 的 weight=1 / bias=0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """x: 形如 [feat_256, feat_512, feat_1024] 的特征列表。"""
        proj_a = self.conv_a(x[0])
        proj_b = self.conv_b(x[1])
        proj_c = self.conv_c(x[2])
        return [proj_a, proj_b, proj_c]


class Projector_Amply_upsample(torch.nn.Module):
    """放大型（Amply）投影器：先把通道数扩大 2 倍再压回原通道数。

    作用：在保持空间尺寸不变的同时，通过"瓶颈-扩张-压缩"的方式学习更丰富的跨模态映射，
    常用于解码阶段的 skip 融合（向解码特征里注入辅助模态信息）。
    结构：3x3 Conv(原通道) -> 1x1 Conv(扩张 2×) -> 1x1 Conv(压回)
    """

    def __init__(self):
        super(Projector_Amply_upsample, self).__init__()

        # 256 -> 512 -> 256
        self.conv_a = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 512, kernel_size=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 256, kernel_size=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
        )
        # 512 -> 1024 -> 512
        self.conv_b = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 1024, kernel_size=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(1024, 512, kernel_size=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
        )
        # 1024 -> 2048 -> 1024
        self.conv_c = torch.nn.Sequential(
            torch.nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(1024, 2048, kernel_size=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(2048),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(2048, 1024, kernel_size=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU(inplace=True),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        proj_a = self.conv_a(x[0])
        proj_b = self.conv_b(x[1])
        proj_c = self.conv_c(x[2])
        return [proj_a, proj_b, proj_c]


class NormalProjector_8(torch.nn.Module):
    """大感受野投影器：通过多次 2×2/stride=2 的下采样再对称反卷积上采样。

    设计动机：扩大感受野并在低分辨率特征空间学习更全局的跨模态对齐关系，
    最后还原到原始分辨率，用于瓶颈阶段的"filter"融合。

    注意：三条支路的下采样次数不同（a 支路最深，c 支路最浅），
    因为不同尺度的输入分辨率不一样，保证输出分辨率与输入相同。
    """

    def __init__(self):
        super(NormalProjector_8, self).__init__()

        # 支路 a：输入 256 通道 / 高分辨率，下采样 8 倍后再逐级上采样回原分辨率
        self.conv_a = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, kernel_size=2, stride=2, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 1024, kernel_size=2, stride=2, bias=False),
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(1024, 2048, kernel_size=2, stride=2, bias=False),
            torch.nn.BatchNorm2d(2048),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2, bias=False),
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
        )
        # 支路 b：输入 512 通道 / 中分辨率，下采样 4 倍
        self.conv_b = torch.nn.Sequential(
            torch.nn.Conv2d(512, 1024, kernel_size=2, stride=2, bias=False),
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(1024, 2048, kernel_size=2, stride=2, bias=False),
            torch.nn.BatchNorm2d(2048),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2, bias=False),
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
        )
        # 支路 c：输入 1024 通道 / 低分辨率，下采样 2 倍
        self.conv_c = torch.nn.Sequential(
            torch.nn.Conv2d(1024, 2048, kernel_size=2, stride=2, bias=False),
            torch.nn.BatchNorm2d(2048),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2, bias=False),
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU(inplace=True)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        proj_a = self.conv_a(x[0])
        proj_b = self.conv_b(x[1])
        proj_c = self.conv_c(x[2])
        # 返回两份相同的投影：第一份给瓶颈 BN；第二份历史上用于其它监督/调试，当前两者等价。
        return [proj_a, proj_b, proj_c], [proj_a, proj_b, proj_c]

