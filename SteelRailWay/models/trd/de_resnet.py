# -*- coding: utf-8 -*-
"""
Reverse-ResNet（反向 ResNet）实现 —— 解码器骨干。

本文件与 ``resnet.py`` 结构对称，不同点在于：
    - 每个 stage 的第一个卷积换成 2x2 转置卷积（``deconv2x2``）做上采样；
    - 通道数从深到浅逐级 **减少**（512 -> 256 -> 128 -> 64），
      与编码器的 layer3 -> layer2 -> layer1 精确对齐；
    - ``_forward_impl`` 返回的特征列表也与编码器输出一一对应，方便做逐层蒸馏/对比。

另外还提供 ``ResNet_Skip``：带可学习权重 skip 融合的解码器，
用于双模态（RGB + 辅助模态）场景，通过 softmax 形式的权重把解码特征与
来自 ``projector_amply`` 的辅助投影混合起来。
"""

import torch
from torch import Tensor
import torch.nn as nn
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional
from torch.nn import functional as F


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


# 预训练权重下载地址（实际使用时 pretrained=False，因为反向 ResNet 结构和标准 ResNet 不完全一致）
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
    """3x3 卷积（带 padding，保持或下采样分辨率）。"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 卷积，用于通道数变换。"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def deconv2x2(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """2x2 转置卷积（用于 2× 上采样）。在反向 ResNet 中替代标准 ResNet 的 stride=2 下采样。"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=2, stride=stride,
                              groups=groups, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    """反向 BasicBlock：结构与标准 BasicBlock 对称。

    stride == 2 时 ``conv1`` 改为 2x2 转置卷积做 **上采样**；
    其余情况仍用 3x3 常规卷积保持分辨率。
    ``upsample`` 分支用于对齐残差旁路的尺寸/通道。
    """
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        upsample: Optional[nn.Module] = None,
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
        # stride=2 -> 上采样；stride=1 -> 常规 3x3
        if stride == 2:
            self.conv1 = deconv2x2(inplanes, planes, stride)
        else:
            self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            identity = self.upsample(x)  # 上采样残差旁路以对齐尺寸/通道

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """反向 Bottleneck：在需要上采样的 stage 把中间的 3x3 换成 2x2 转置卷积。"""
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        upsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        # 关键差别：stride=2 时用转置卷积实现上采样
        if stride == 2:
            self.conv2 = deconv2x2(width, width, stride, groups, dilation)
        else:
            self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
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

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """反向 ResNet 主体：三段 stage 依次把特征上采样 2 倍并减少通道数。

    通道流（以 Wide-ResNet50_2 镜像为例）：
        输入（瓶颈特征）: 2048 通道 / 8x8
        layer1: 2048 -> 1024 / 16x16
        layer2: 1024 -> 512  / 32x32
        layer3: 512  -> 256  / 64x64
    输出按 **从浅到深** 顺序排列（即 [feat_c, feat_b, feat_a]），与编码器特征一一对应。
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

        # 起始通道最大，之后逐层减少
        self.inplanes = 512 * block.expansion
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        # 三段 stage，每段都做一次 2× 上采样（stride=2）
        self.layer1 = self._make_layer(block, 256, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        """构建一段反向 stage：第一个 block 做上采样，其余维持分辨率。"""
        norm_layer = self._norm_layer
        upsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            # 残差旁路用转置卷积对齐
            upsample = nn.Sequential(
                deconv2x2(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, upsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # 尺寸标注以 Wide-ResNet50_2 为例
        feature_a = self.layer1(x)           # 2048x8x8 -> 1024x16x16
        feature_b = self.layer2(feature_a)   # 1024x16x16 -> 512x32x32
        feature_c = self.layer3(feature_b)   # 512x32x32 -> 256x64x64

        # 注意返回顺序：从"最浅层对应的解码特征"到"最深层"，与编码器 [feat_a, feat_b, feat_c] 一一对应
        return [feature_c, feature_b, feature_a]

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
    """反向 ResNet 构造工具。pretrained 一般为 False（反向网络结构与原 ResNet 不一致）。"""
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)

        model.load_state_dict(state_dict)
    return model


# -------- 下面是各架构的便捷构造函数，和 resnet.py 一一对应 --------

def de_resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """反向 ResNet-18。"""
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def de_resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """反向 ResNet-34。"""
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def de_resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """反向 ResNet-50。"""
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """反向 ResNet-101。"""
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """反向 ResNet-152。"""
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """反向 ResNeXt-50 32x4d。"""
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """反向 ResNeXt-101 32x8d。"""
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def de_wide_resnet50_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """反向 Wide-ResNet50_2：本项目单模态解码器的默认选择。"""
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def de_wide_resnet101_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """反向 Wide-ResNet101_2。"""
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


class ResNet_Skip(nn.Module):
    """带可学习 skip 融合的反向 ResNet（用于双模态场景）。

    与 ``ResNet`` 相比，本类在 ``forward_fuse`` 中每一级解码后都会把
    "解码结果"与"辅助模态投影（x_assit，来自 projector_amply）"按
    softmax 归一化的可学习权重做加权混合，从而让解码路径始终注入辅助模态信息。

    权重 ``w_a / w_b / w_c`` 各为 2 维参数，经 softmax 后得到两权重 w1 + w2 = 1。
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
        super(ResNet_Skip, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 512 * block.expansion
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        # 三段反向 stage，与 ResNet 一致
        self.layer1 = self._make_layer(block, 256, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                          dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2,
                                          dilate=replace_stride_with_dilation[1])

        # 每个尺度上用来做 softmax 融合的可学习权重（2 维：解码特征 vs 辅助模态投影）
        self.w_a = torch.nn.Parameter(torch.tensor([1.0, 1.0]), requires_grad=True)
        self.w_b = torch.nn.Parameter(torch.tensor([1.0, 1.0]), requires_grad=True)
        self.w_c = torch.nn.Parameter(torch.tensor([1.0, 1.0]), requires_grad=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        """和 ResNet._make_layer 相同。"""
        norm_layer = self._norm_layer
        upsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                deconv2x2(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, upsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _make_layer_2x(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        """通道数加倍版本的 _make_layer：历史实验中尝试将解码与辅助投影
        concat 后再卷积，故第一个 block 的输入通道是 self.inplanes * 2。当前版本未使用。"""
        norm_layer = self._norm_layer
        upsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                deconv2x2(self.inplanes*2, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes*2, planes, stride, upsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward_fuse(self, x, x_assit, x_encoder=None, proj_filter=None, disable_ca: bool = False):
        """带 skip 融合的前向。

        参数：
            x        : 瓶颈特征（来自 BN_fuse_layer 的输出）
            x_assit  : 辅助模态投影列表（projector_amply 输出，顺序与编码器特征一致，
                       即 [proj_a_256, proj_b_512, proj_c_1024]）
            x_encoder, proj_filter : 预留参数，当前未启用（历史实验用于基于
                       余弦相似度的动态权重融合）
            disable_ca : 推理期消融开关；为 True 时跳过解码阶段辅助 skip 融合，
                         仅使用纯解码路径，保持已训练权重和张量形状不变。
        返回：
            融合后的解码特征列表 [feat_c, feat_b, feat_a]   ——用于与 Encoder 对齐
            未融合的纯解码特征列表 [feat_c_0, feat_b_0, feat_a_0] ——供多任务监督
        """
        # 第一级：解码深层，融合最深的辅助投影 x_assit[2]
        feature_a_0 = self.layer1(x)
        if disable_ca:
            feature_a = feature_a_0
        else:
            w1 = torch.exp(self.w_a[0]) / torch.sum(torch.exp(self.w_a))  # softmax 权重 1
            w2 = torch.exp(self.w_a[1]) / torch.sum(torch.exp(self.w_a))  # softmax 权重 2
            feature_a = w1 * feature_a_0 + w2 * x_assit[2]

        # 第二级：解码中层，融合 x_assit[1]
        feature_b_0 = self.layer2(feature_a)
        if disable_ca:
            feature_b = feature_b_0
        else:
            w1 = torch.exp(self.w_b[0]) / torch.sum(torch.exp(self.w_b))
            w2 = torch.exp(self.w_b[1]) / torch.sum(torch.exp(self.w_b))
            feature_b = w1 * feature_b_0 + w2 * x_assit[1]

        # 第三级：解码浅层，融合 x_assit[0]
        feature_c_0 = self.layer3(feature_b)
        if disable_ca:
            feature_c = feature_c_0
        else:
            w1 = torch.exp(self.w_c[0]) / torch.sum(torch.exp(self.w_c))
            w2 = torch.exp(self.w_c[1]) / torch.sum(torch.exp(self.w_c))
            feature_c = w1 * feature_c_0 + w2 * x_assit[0]

        # ===== 下方大段注释为历史实验：基于 Encoder-解码 / Encoder-辅助的
        # 余弦相似度动态生成权重，效果不如上面的 softmax 学习权重，故保留注释做参考 =====
        # with torch.no_grad():
        #     sim_a_c = torch.unsqueeze(F.cosine_similarity(x_encoder[0], x_assit[0]), dim=1)
        #     sim_a_b = torch.unsqueeze(F.cosine_similarity(x_encoder[1], x_assit[1]), dim=1)
        #     sim_a_a = torch.unsqueeze(F.cosine_similarity(x_encoder[2], x_assit[2]), dim=1)
        # feature_a_0 = self.layer1(x)
        # with torch.no_grad():
        #     sim_a = torch.unsqueeze(F.cosine_similarity(x_encoder[2], feature_a_0), dim=1)
        #     dis_a_de = 1 - sim_a
        #     dis_a_am = 1 - sim_a_a
        # dis_a_de = torch.relu(self.w_a[0]) * dis_a_de
        # dis_a_am = torch.relu(self.w_a[1]) * dis_a_am
        # feature_a_1 = dis_a_de.detach() / (dis_a_de.detach() + dis_a_am.detach()) * feature_a_0 + dis_a_am.detach() / (dis_a_de.detach() + dis_a_am.detach()) * x_assit[2]
        # feature_a = dis_a_de / (dis_a_de + dis_a_am) * feature_a_0.detach() + dis_a_am / (
        #             dis_a_de + dis_a_am) * x_assit[2].detach()
        # feature_b_0 = self.layer2(feature_a_1)
        # with torch.no_grad():
        #     sim_b = torch.unsqueeze(F.cosine_similarity(x_encoder[1], feature_b_0), dim=1)
        #     dis_b_de = 1 - sim_b
        #     dis_b_am = 1 - sim_a_b
        # dis_b_de = torch.relu(self.w_b[0]) * dis_b_de
        # dis_b_am = torch.relu(self.w_b[1]) * dis_b_am
        # feature_b_1 = dis_b_de.detach() / (dis_b_de.detach() + dis_b_am.detach()) * feature_b_0 + dis_b_am.detach() / (dis_b_de.detach() + dis_b_am.detach()) * x_assit[1]
        # feature_b = dis_b_de / (dis_b_de + dis_b_am) * feature_b_0.detach() + dis_b_am / (dis_b_de + dis_b_am) * \
        #             x_assit[1].detach()
        # feature_c_0 = self.layer3(feature_b_1)
        # with torch.no_grad():
        #     sim_c = torch.unsqueeze(F.cosine_similarity(x_encoder[0], feature_c_0), dim=1)
        #     dis_c_de = 1 - sim_c
        #     dis_c_am = 1 - sim_a_c
        # dis_c_de = torch.relu(self.w_c[0]) * dis_c_de
        # dis_c_am = torch.relu(self.w_c[1]) * dis_c_am
        # feature_c = dis_c_de / (dis_c_de + dis_c_am) * feature_c_0.detach() + dis_c_am / (dis_c_de + dis_c_am) * x_assit[0].detach()

        # 返回按编码器顺序排列的两组特征：融合版 + 纯解码版
        return [feature_c, feature_b, feature_a], [feature_c_0, feature_b_0, feature_a_0]


def _resnet_skip(
        arch: str,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        pretrained: bool,
        progress: bool,
        **kwargs: Any
) -> ResNet:
    """带 skip 融合的反向 ResNet 构造工具。"""
    model = ResNet_Skip(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)

        model.load_state_dict(state_dict)
    return model

def de_wide_resnet50_2_skip(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """双模态解码器默认骨干：带 softmax 权重 skip 融合的反向 Wide-ResNet50_2。"""
    kwargs['width_per_group'] = 64 * 2
    return _resnet_skip('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)
