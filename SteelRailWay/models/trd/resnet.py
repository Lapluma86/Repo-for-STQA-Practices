# -*- coding: utf-8 -*-
"""
ResNet 编码器实现（改写自 torchvision 官方版）。

本文件在 torchvision 标准 ResNet 基础上做了两项本项目特有的改造：

1. ``ResNet._forward_impl`` 只返回 layer1/layer2/layer3 的 **多尺度特征**（未使用 layer4 和 fc），
   因为 TRD 异常检测框架需要的是多层次特征图而不是分类 logits。
2. 增加了两个自定义瓶颈融合模块：
   - ``BN_layer``：单模态瓶颈，把 3 个尺度的特征（layer1/2/3 输出）通过 stride=2 卷积对齐到
     同一分辨率后 concat，再经过若干 AttnBottleneck 生成最终的瓶颈特征给解码器。
   - ``BN_fuse_layer``：双模态瓶颈，在上述基础上把主模态（RGB）特征与辅助模态投影（depth/normal）
     先 concat 再做卷积对齐，实现跨模态融合。

同时提供了构造便捷函数：``wide_resnet50_2``、``bn``、``bn_fuse`` 等。
"""

import torch
from torch import Tensor
import torch.nn as nn
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    # 老版本 torch 的兼容导入
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


# ImageNet 官方预训练权重下载地址
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 卷积（默认带 padding 保持分辨率，可选 stride 下采样 / dilation 膨胀）。"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 卷积，常用于通道变换和残差分支的降采样对齐。"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """ResNet-18/34 使用的基础残差块：两层 3x3 卷积 + 残差直连。"""
    expansion: int = 1  # 输出通道 = planes * expansion

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # 当 stride!=1 时，self.conv1 承担下采样，残差旁路也要通过 downsample 对齐尺寸
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x  # 保留原始输入用于残差

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)  # 尺寸 / 通道对齐

        out += identity  # 残差相加
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """ResNet-50/101/152 使用的瓶颈残差块：1x1 降维 -> 3x3 -> 1x1 升维 -> 残差。

    说明（ResNet V1.5）：
    torchvision 把 3x3 的 stride 放在中间（conv2），原论文放在 conv1，V1.5 精度略高。
    """

    expansion: int = 4  # 输出通道 = planes * 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Wide-ResNet 里 base_width=128，此处 width 会翻倍，从而加宽中间通道
        width = int(planes * (base_width / 64.)) * groups
        # conv2 承担下采样（stride），downsample 分支同步对齐
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """标准 ResNet 骨干网络。

    与 torchvision 的主要区别：本版 ``_forward_impl`` 返回 layer1/layer2/layer3 的
    多尺度特征列表，供异常检测的 teacher/student 对齐使用；不做全局池化和分类头。
    """

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64  # 第一阶段的输入通道
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # 每个元素表示对应 stage 是否用空洞卷积代替 stride=2 下采样
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        # Stem：7x7 conv -> BN -> ReLU -> 3x3 maxpool，输入 RGB 3 通道，下采样 4 倍
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 四个 stage：每个 stage 通道翻倍、分辨率减半（除 layer1）
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 权重初始化：Conv 用 Kaiming，BN weight=1 / bias=0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-init residual：让最后一个 BN 的 weight=0，使残差分支起步等价于恒等映射
        # 在 ImageNet 上能带来 0.2~0.3% 精度提升（见 https://arxiv.org/abs/1706.02677）
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        """构建一个 stage：第一个 block 可能带下采样，其余 block 保持分辨率。"""
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            # 用 dilation 代替 stride，保留分辨率
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            # 残差旁路通过 1x1 conv 对齐通道/分辨率
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        # 第一个 block 负责下采样
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        # 后续 block 维持分辨率
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # stem 阶段
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 多尺度特征提取：只取前三层输出，不走 layer4/fc
        feature_a = self.layer1(x)           # 256 通道（假设 Wide-ResNet50_2）
        feature_b = self.layer2(feature_a)   # 512 通道
        feature_c = self.layer3(feature_b)   # 1024 通道
        # feature_d = self.layer4(feature_c)  # 本框架不使用最深层特征

        return [feature_a, feature_b, feature_c]

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    """通用 ResNet 构造函数：按 arch 名可加载对应 ImageNet 预训练权重。"""
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


class AttnBasicBlock(nn.Module):
    """带 attention 开关的 BasicBlock（本项目瓶颈层使用）。

    结构与标准 BasicBlock 相同；``attention`` 参数作为历史实验开关保留，
    当前版本未在 forward 中启用显式注意力（行为与普通 BasicBlock 一致）。
    """
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        attention: bool = True,
    ) -> None:
        super(AttnBasicBlock, self).__init__()
        self.attention = attention
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class AttnBottleneck(nn.Module):
    """带 attention 开关的 Bottleneck（Wide-ResNet50_2 对应的瓶颈融合层使用）。

    行为与标准 Bottleneck 一致；``attention`` 为历史实验开关。
    """

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        attention: bool = True,
    ) -> None:
        super(AttnBottleneck, self).__init__()
        self.attention = attention
        #print("Attention:",self.attention)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BN_layer(nn.Module):
    """单模态瓶颈融合层（OCBE 的简化版）。

    功能：把 Encoder 输出的三个尺度特征 [feat_a, feat_b, feat_c]
    通过下采样对齐到同一尺寸再 concat，最后经过若干 AttnBottleneck 生成解码器的起点特征。

    维度（以 Wide-ResNet50_2 为例）：
        feat_a : 256 通道，较大分辨率
        feat_b : 512 通道，中分辨率
        feat_c : 1024 通道，较小分辨率
        - feat_a 经两次 stride=2 卷积下采样 4 倍 -> 1024 通道
        - feat_b 经一次 stride=2 卷积下采样 2 倍 -> 1024 通道
        - 三者 concat 后通道变为 3072，再经 bn_layer 得到 2048 通道的瓶颈特征
    """

    def __init__(self,
                 block: Type[Union[BasicBlock, Bottleneck]],
                 layers: int,
                 groups: int = 1,
                 width_per_group: int = 64,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 ):
        super(BN_layer, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.groups = groups
        self.base_width = width_per_group
        self.inplanes = 256 * block.expansion
        self.dilation = 1
        # 主体：stride=2，把 concat 后的特征进一步压缩/深化为解码器的输入
        self.bn_layer = self._make_layer(block, 512, layers, stride=2)

        # 对 feat_a（最浅层）的两次下采样：把其分辨率对齐到 feat_c
        self.conv1 = conv3x3(64 * block.expansion, 128 * block.expansion, 2)
        self.bn1 = norm_layer(128 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(128 * block.expansion, 256 * block.expansion, 2)
        self.bn2 = norm_layer(256 * block.expansion)
        # 对 feat_b 的一次下采样
        self.conv3 = conv3x3(128 * block.expansion, 256 * block.expansion, 2)
        self.bn3 = norm_layer(256 * block.expansion)

        # 备用 1x1 conv（当前未在 forward 中使用，保留以兼容早期版本权重）
        self.conv4 = conv1x1(1024 * block.expansion, 512 * block.expansion, 1)
        self.bn4 = norm_layer(512 * block.expansion)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            # 注意这里 inplanes * 3，对应 forward 中三个尺度 concat 后的通道数
            downsample = nn.Sequential(
                conv1x1(self.inplanes*3, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes*3, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # x = [feat_a, feat_b, feat_c]
        l1 = self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x[0]))))))  # feat_a 下采样到 feat_c 同分辨率
        l2 = self.relu(self.bn3(self.conv3(x[1])))                                    # feat_b 下采样到 feat_c 同分辨率
        feature = torch.cat([l1, l2, x[2]], 1)                                        # 通道维 concat
        output = self.bn_layer(feature)                                               # 统一卷积到瓶颈特征
        return output.contiguous()

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


class BN_fuse_layer(nn.Module):
    """双模态瓶颈融合层。

    相较 ``BN_layer``，在每个尺度上先把主模态（RGB）特征与辅助模态投影
    （projector 输出）按通道 concat，再走下采样对齐，最后再做三尺度 concat 与卷积融合。
    这让瓶颈处同时拿到两种模态的信息。
    """

    def __init__(self,
                 block: Type[Union[BasicBlock, Bottleneck]],
                 layers: int,
                 groups: int = 1,
                 width_per_group: int = 64,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 ):
        super(BN_fuse_layer, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.groups = groups
        self.base_width = width_per_group
        self.inplanes = 256 * block.expansion
        self.dilation = 1
        self.bn_layer = self._make_layer(block, 512, layers, stride=2)

        # 各尺度的 "融合 + 下采样"：输入通道 = 原通道 * 2（因为 RGB 与辅助投影 concat）
        self.conv1 = conv3x3(64 * block.expansion*2, 128 * block.expansion*2, 2)
        self.bn1 = norm_layer(128 * block.expansion*2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(128 * block.expansion*2, 256 * block.expansion, 2)
        self.bn2 = norm_layer(256 * block.expansion)

        self.conv3 = conv3x3(128 * block.expansion * 2, 256 * block.expansion, 2)
        self.bn3 = norm_layer(256 * block.expansion)

        # 最深尺度不需要再下采样，只做通道对齐（原通道*2 -> 原通道）
        self.conv4 = conv3x3(256 * block.expansion * 2, 256 * block.expansion, 1)
        self.bn4 = norm_layer(256 * block.expansion)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes*3, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes*3, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x, x_assit):
        """x: 主模态特征列表；x_assit: 辅助模态投影列表。两者按通道 concat 融合。"""
        l1 = self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(torch.cat([x[0], x_assit[0]], 1)))))))  # 浅层
        l2 = self.relu(self.bn3(self.conv3(torch.cat([x[1], x_assit[1]], 1))))                                    # 中层
        l3 = self.relu(self.bn4(self.conv4(torch.cat([x[2], x_assit[2]], 1))))                                    # 深层
        # 以下为历史上尝试过的"逐元素相加"融合方案，当前改为 concat
        # l1 = self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x[0] + x_assit[0]))))))
        # l2 = self.relu(self.bn3(self.conv3(x[1] + x_assit[1])))
        # l3 = self.relu(self.bn4(self.conv4(x[2] + x_assit[2])))
        feature = torch.cat([l1, l2, l3], 1)  # 三尺度 concat
        output = self.bn_layer(feature)
        return output.contiguous()

    def forward(self, x, x_assit):
        return self._forward_impl(x, x_assit)


# ======================== 下面是便捷构造函数 ========================
# 多数直接调用 _resnet，拷贝自 torchvision，用于加载预训练权重

def resnet18(pretrained: bool = False, progress: bool = True,**kwargs: Any) -> ResNet:
    r"""ResNet-18。详见 https://arxiv.org/pdf/1512.03385.pdf"""
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)

def bn_resnet18(**kwargs: Any):
    """与 resnet18 配套的瓶颈层（用于异常检测管线）。"""
    return BN_layer(AttnBasicBlock,2,**kwargs)


def resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-34，同时返回配套的 BN_layer。"""
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs), BN_layer(AttnBasicBlock,3,**kwargs)


def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50，同时返回配套的 BN_layer。"""
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs), BN_layer(AttnBottleneck,3,**kwargs)


def resnet101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-101，同时返回配套的 BN_layer。"""
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs), BN_layer(AttnBasicBlock,3,**kwargs)


def resnet152(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-152，同时返回配套的 BN_layer。"""
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs), BN_layer(AttnBottleneck,3,**kwargs)


def resnext50_32x4d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-50 32x4d。"""
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-101 32x8d。"""
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-101-2。"""
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs), BN_layer(AttnBottleneck,3,**kwargs)


def wide_resnet50_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-50-2：本项目编码器默认选择。bottleneck 中间通道加倍，特征更丰富。"""
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def bn(**kwargs: Any):
    """构造单模态瓶颈层（搭配 wide_resnet50_2 使用）。"""
    kwargs['width_per_group'] = 64 * 2
    return BN_layer(AttnBottleneck, 3, **kwargs)


def bn_fuse(**kwargs: Any):
    """构造双模态瓶颈融合层（搭配 wide_resnet50_2 + 辅助模态 Projector 使用）。"""
    kwargs['width_per_group'] = 64 * 2
    return BN_fuse_layer(AttnBottleneck, 3, **kwargs)

