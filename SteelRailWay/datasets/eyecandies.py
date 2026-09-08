# -*- coding: utf-8 -*-
"""
Eyecandies 数据集（含 RGB + Depth + Normals 三路输入）加载器。

Eyecandies 是另一个 3D 异常检测基准。与 MVTec 3D-AD 的区别：
    - 目录里除了 rgb 和 depth，还直接提供预先计算好的 normals 图像，
      因此本加载器 **不用再做 Sobel 计算法向量**，直接读取 PNG 即可。
    - 当前实现里 ``depth_img = normals_img``，即真正送入网络的辅助模态是 normals；
      depth 路径仅在 data_info 里保留，未被 __getitem__ 使用（便于日后切换）。

提供两类：
    - ``EyeRGBDDataset``       : 常规训练/评估。
    - ``EyeRGBDDataset_test``  : 额外返回 rgb_path，方便可视化保存。
"""

from torch.utils.data import Dataset
import os
from PIL import Image


from .geo_utils import *  # 提供 np / torch / cv2 等别名


class EyeRGBDDataset(Dataset):
    """Eyecandies 训练/评估数据集。辅助模态使用 normals（由作者预计算好的 PNG）。

    参数（保持与 MVTec 版本签名一致，但 depth_transform/test_min/test_max 当前未用）：
        data_dir        : 数据根目录
        transform       : RGB 与 normals 公用的 transform
        depth_transform : 预留参数（当前 normals 已复用 transform，故未使用）
        test            : 预留参数
        gt_transform    : gt 掩码的 transform
    """

    def __init__(self, data_dir, transform=None, depth_transform=None, test=False, gt_transform=None, test_min=0.0, test_max=1.0):
        self.transform = transform
        self.gt_transform = gt_transform
        self.data_info = self.get_data_info(data_dir)
        self.depth_transform = depth_transform

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, index):
        rgb_path, depth_path, normals_path, gt, ad_label, ad_type = self.data_info[index]

        # 读 RGB
        rgb_img = Image.open(rgb_path).convert('RGB')
        if self.transform is not None:
            rgb_img = self.transform(rgb_img)

        # 读 normals 并沿用相同 transform；depth_img 直接指向 normals_img（当前辅助模态 = normals）
        normals_img = Image.open(normals_path).convert('RGB')
        if self.transform is not None:
            normals_img = self.transform(normals_img)
        depth_img = normals_img  # 保持与外部接口统一的变量名

        # 正常样本无 gt
        if gt == 0:
            gt = torch.zeros([1, rgb_img.size()[-2], rgb_img.size()[-2]])
        else:
            gt = Image.open(gt).convert('L')
            if self.gt_transform is not None:
                gt = self.gt_transform(gt)

        assert rgb_img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return rgb_img, depth_img, gt, ad_label, ad_type

    def get_data_info(self, data_dir):
        """扫描 Eyecandies 目录收集样本元信息列表。

        期望目录结构：
            data_dir/<type>/{rgb, depth, normals, gt}/*.png
        "good" 子目录为正常样本。
        """
        data_info = list()

        for root, dirs, _ in os.walk(data_dir):
            for sub_dir in dirs:
                rgb_names = os.listdir(os.path.join(root, sub_dir, 'rgb'))
                rgb_names = list(filter(lambda x: x.endswith('.png'), rgb_names))
                for rgb_name in rgb_names:
                    rgb_path = os.path.join(root, sub_dir, 'rgb', rgb_name)
                    depth_path = os.path.join(root, sub_dir, 'depth', rgb_name)
                    normals_path = os.path.join(root, sub_dir, 'normals', rgb_name)
                    if sub_dir == 'good':
                        data_info.append((rgb_path, depth_path, normals_path, 0, 0, sub_dir))
                    else:
                        gt_name = rgb_name
                        gt_path = os.path.join(root, sub_dir, 'gt', gt_name)
                        data_info.append((rgb_path, depth_path, normals_path, gt_path, 1, sub_dir))

            break  # 只遍历第一层子目录

        np.random.shuffle(data_info)

        return data_info


class EyeRGBDDataset_test(Dataset):
    """测试可视化专用：与 ``EyeRGBDDataset`` 一致，但额外返回 rgb_path。"""

    def __init__(self, data_dir, transform=None, depth_transform=None, test=False, gt_transform=None, test_min=0.0, test_max=1.0):
        # gt_dir == None --> train；gt_dir != None --> test（作者原注释）
        self.transform = transform
        self.gt_transform = gt_transform
        self.data_info = self.get_data_info(data_dir)
        self.depth_transform = depth_transform

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, index):
        rgb_path, depth_path, normals_path, gt, ad_label, ad_type = self.data_info[index]
        rgb_img = Image.open(rgb_path).convert('RGB')
        if self.transform is not None:
            rgb_img = self.transform(rgb_img)

        normals_img = Image.open(normals_path).convert('RGB')
        if self.transform is not None:
            normals_img = self.transform(normals_img)
        depth_img = normals_img

        if gt == 0:
            gt = torch.zeros([1, rgb_img.size()[-2], rgb_img.size()[-2]])
        else:
            gt = Image.open(gt).convert('L')
            if self.gt_transform is not None:
                gt = self.gt_transform(gt)

        assert rgb_img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        # 比训练版本多返回 rgb_path，方便写可视化结果时获取文件名
        return rgb_img, depth_img, gt, ad_label, ad_type, rgb_path

    def get_data_info(self, data_dir):
        """与 ``EyeRGBDDataset.get_data_info`` 相同。"""
        data_info = list()

        for root, dirs, _ in os.walk(data_dir):
            for sub_dir in dirs:
                rgb_names = os.listdir(os.path.join(root, sub_dir, 'rgb'))
                rgb_names = list(filter(lambda x: x.endswith('.png'), rgb_names))
                for rgb_name in rgb_names:
                    rgb_path = os.path.join(root, sub_dir, 'rgb', rgb_name)
                    depth_path = os.path.join(root, sub_dir, 'depth', rgb_name)
                    normals_path = os.path.join(root, sub_dir, 'normals', rgb_name)
                    if sub_dir == 'good':
                        data_info.append((rgb_path, depth_path, normals_path, 0, 0, sub_dir))
                    else:
                        gt_name = rgb_name
                        gt_path = os.path.join(root, sub_dir, 'gt', gt_name)
                        data_info.append((rgb_path, depth_path, normals_path, gt_path, 1, sub_dir))

            break

        np.random.shuffle(data_info)

        return data_info

