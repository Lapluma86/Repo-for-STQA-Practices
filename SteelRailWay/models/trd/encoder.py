# -*- coding: utf-8 -*-
"""
编码器（Encoder）模块定义文件。
本文件提供基于 Wide-ResNet50_2 的图像特征提取器，用于将输入图片
编码成多尺度的深层特征，供后续解码器 / 异常检测逻辑使用。
"""

import torch
from .resnet import wide_resnet50_2  # 自定义版本的 Wide-ResNet50_2（支持多层特征输出）


def init_weight(m):
    """通用的权重初始化函数。

    - 对线性层和卷积层使用 Xavier 正态初始化，有助于训练稳定。
    - 在需要时可通过 ``model.apply(init_weight)`` 统一应用到整个模型上。
    """
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)


class ResNet50Encoder(torch.nn.Module):
    """基于 Wide-ResNet50_2 的编码器封装。

    作用：调用 ``wide_resnet50_2`` 从输入 RGB 图像抽取多层次特征（通常为 layer1 / layer2 / layer3 的输出），
    在 TRD 异常检测框架中，这些多尺度特征会与解码器重建结果做对比来判定异常。

    参数：
        pretrained (bool): 是否加载 ImageNet 预训练权重。训练时一般设为 True 以获得更好的初始特征表达。
    """

    def __init__(self, pretrained=True):
        super(ResNet50Encoder, self).__init__()

        # 用自定义的 wide_resnet50_2（见 models/resnet.py），返回多尺度特征列表
        self.encoder = wide_resnet50_2(pretrained=pretrained)

    def forward(self, x):
        # 注释掉的 no_grad 表示历史上曾尝试冻结 Encoder，但当前版本允许其参数参与梯度更新
        # with torch.no_grad():
        #     x = self.encoder(x)
        x = self.encoder(x)  # 返回形如 [feat1, feat2, feat3] 的多尺度特征

        return x

