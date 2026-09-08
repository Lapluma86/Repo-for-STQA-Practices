# -*- coding: utf-8 -*-
"""
MVTec 3D-AD 数据集 RGB + Depth（RGBD）加载器。

与 ``dataset_rgbn.py`` 辅助模态用法向图不同，本文件辅助模态直接使用
深度图（depth）。主要工作：
    1. 遍历 ``data_dir`` 下每个缺陷类型子目录，收集 (rgb_path, depth_path, gt, label, type)。
    2. 读取深度图后做：平面分割 → 归一化到 [0.1, 0.9] → 缺失像素填充。
    3. 训练期遍历整体数据估计全局最小/最大深度（带 10% 余量）。
    4. __getitem__ 返回：rgb_img, depth_img(3×), gt, ad_label(int), ad_type(str)。

两类：
    - ``MVTecADRGBDDataset`` : 训练/常规测试使用。
    - ``MVTecADRGBDDataset_test`` : 返回值额外包含文件路径和可视化缩略图，
      便于可视化分析（当前仅 test 流程使用）。
"""

from torch.utils.data import Dataset
import os
from PIL import Image

from .geo_utils import *  # 提供 np、cv2、torch、fill_depth_map、get_plane_mask 等工具

import tifffile  # 读取 .tiff 深度图


def get_max_min_depth_img(image_path):
    """读取单张 .tiff 深度（xyz）图，返回该图 z 通道的最小/最大有效值。

    约定：深度图为 (H, W, 3) 的 xyz 格式；第三通道 (z) 为深度，值为 0 表示缺失。
    返回的最小值排除了 0 像素，用于后续全局归一化。
    """
    image = tifffile.imread(image_path).astype(np.float32)
    image_t = (
        np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32)
    )
    image = image_t[:, :, 2]  # 取 z 通道作为深度
    zero_mask = np.where(image == 0, np.ones_like(image), np.zeros_like(image))
    im_max = np.max(image)
    # 把 0 像素替换成一个大值（1000），求 min 时就自动忽略它们
    im_min = np.min(image * (1.0 - zero_mask) + 1000 * zero_mask)
    return im_min, im_max


class MVTecADRGBDDataset(Dataset):
    """RGBD 训练/测试数据集主类。

    参数：
        data_dir        : 数据根目录（下属子目录为缺陷类型）。
        transform       : RGB 图像的 torchvision transform。
        depth_transform : 对深度图（被复制为 3 通道以匹配 RGB）做的 transform。
        test            : True 时跳过全局 min/max 估计，使用传入的 test_min/test_max。
        gt_transform    : Ground-truth 掩码的 transform。
        test_min/test_max : 测试期使用的全局深度范围，通常从训练集统计后传入保持一致。
    """

    def __init__(self, data_dir, transform=None, depth_transform=None, test=False, gt_transform=None, test_min=0.0, test_max=1.0):
        self.transform = transform
        self.gt_transform = gt_transform
        self.data_info = self.get_data_info(data_dir)  # list of tuples，见 get_data_info
        self.depth_transform = depth_transform

        # 计算/传入全局深度 min/max
        self.global_max, self.global_min = 1, 0
        if not test:
            # 训练阶段：遍历所有样本估计一次
            for data_info_i in self.data_info:
                rgb_path, depth_path, gt, ad_label, ad_type = data_info_i
                im_min, im_max = get_max_min_depth_img(depth_path)
                self.global_min = min(self.global_min, im_min)
                self.global_max = max(self.global_max, im_max)
            # 留 10% 余量，防止测试集出现略超范围的值
            self.global_min = self.global_min * 0.9
            self.global_max = self.global_max * 1.1
        else:
            # 测试阶段：沿用训练阶段的全局范围
            self.global_max = test_max
            self.global_min = test_min

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, index):
        rgb_path, depth_path, gt, ad_label, ad_type = self.data_info[index]

        # 读取 RGB 并做增强/归一化
        rgb_img = Image.open(rgb_path).convert('RGB')
        if self.transform is not None:
            rgb_img = self.transform(rgb_img)

        # 读取并处理深度图（返回 1×H×W 的深度 tensor 与前景 mask）
        depth_img, plane_mask = self.get_depth_image(depth_path, rgb_img.size()[-2])

        if self.depth_transform is not None:
            # 把单通道深度复制 3 份以匹配 3 通道输入网络
            depth_img = torch.cat([depth_img, depth_img, depth_img], dim=0)
            depth_img = self.depth_transform(depth_img)

        # 正常样本没有 gt，用全零占位
        if gt == 0:
            gt = torch.zeros([1, rgb_img.size()[-2], rgb_img.size()[-2]])
        else:
            gt = Image.open(gt).convert('L')
            if self.gt_transform is not None:
                gt = self.gt_transform(gt)

        assert rgb_img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return rgb_img, depth_img, gt, ad_label, ad_type

    def get_data_info(self, data_dir):
        """扫描目录收集 (rgb_path, depth_path, gt_path_or_0, label, type) 列表。

        目录结构假设：
            data_dir/
              <type>/
                rgb/*.png
                xyz/*.tiff   (与 rgb 同名)
                gt/*.png     (仅缺陷类型有)
        子目录名为 "good" 视为正常样本（label=0, gt=0）。
        """
        data_info = list()

        # 只遍历第一层子目录（break 确保不递归）
        for root, dirs, _ in os.walk(data_dir):
            for sub_dir in dirs:
                rgb_names = os.listdir(os.path.join(root, sub_dir, 'rgb'))
                rgb_names = list(filter(lambda x: x.endswith('.png'), rgb_names))
                for rgb_name in rgb_names:
                    rgb_path = os.path.join(root, sub_dir, 'rgb', rgb_name)
                    depth_name = rgb_name.replace(".png", ".tiff")
                    depth_path = os.path.join(root, sub_dir, 'xyz', depth_name)
                    if sub_dir == 'good':
                        data_info.append((rgb_path, depth_path, 0, 0, sub_dir))
                    else:
                        gt_name = rgb_name
                        gt_path = os.path.join(root, sub_dir, 'gt', gt_name)
                        data_info.append((rgb_path, depth_path, gt_path, 1, sub_dir))

            break  # 只遍历一层

        # 打乱顺序，避免训练时同类型样本聚集
        np.random.shuffle(data_info)

        return data_info

    def get_depth_image(self, file, size=None):
        """读取 .tiff xyz 深度图 → 单通道归一化深度 + 前景平面掩码。

        处理步骤：
            1. resize 到目标大小（与 RGB 一致）。
            2. 用 RANSAC 估计主平面，生成前景 mask，并把零值（传感器缺失）排除。
            3. 仅保留前景区域，做 min-max 归一化到 [0, 1]，再压到 [0.1, 0.9] 区间，
               目的是保留一个"缺失=0"的特殊值，便于网络区分有效/无效像素。
            4. 用局部均值填充缺失像素（``fill_depth_map``）以消除空洞。
        返回：
            depth_img : (1, H, W) FloatTensor，取值 [0.1, 0.9]（被 fill 过的缺失处是局部均值）
            plane_mask : (H, W) FloatTensor，1=前景 0=背景
        """
        depth_img = tifffile.imread(file).astype(np.float32)
        size = self.image_size if size is None else size
        # 最近邻 resize，避免在缺失边缘引入错误的平均深度值
        depth_img = cv2.resize(
            depth_img, (size, size), 0, 0, interpolation=cv2.INTER_NEAREST
        )
        depth_img = np.array(depth_img)

        image = depth_img
        image_t = (
            np.array(image)
            .reshape((image.shape[0], image.shape[1], 3))
            .astype(np.float32)
        )
        image = image_t[:, :, 2]  # z 通道

        zero_mask = np.where(image == 0, np.ones_like(image), np.zeros_like(image))
        plane_mask = get_plane_mask(image_t)  # 0=背景平面, 1=物体前景
        # 前景 mask 再排除 z=0 的传感器缺失点
        plane_mask[:, :, 0] = plane_mask[:, :, 0] * (1.0 - zero_mask)
        plane_mask = fill_plane_mask(plane_mask)  # 平滑前景 mask

        # 只保留前景深度再做归一化
        image = image * plane_mask[:, :, 0]
        zero_mask = np.where(image == 0, np.ones_like(image), np.zeros_like(image))
        im_min = np.min(image * (1.0 - zero_mask) + 1000 * zero_mask)
        im_max = np.max(image)
        image = (image - im_min) / (im_max - im_min)
        image = image * 0.8 + 0.1  # [0,1] -> [0.1, 0.9]
        image = image * (1.0 - zero_mask)  # 缺失像素保持为 0（特殊值）
        image = fill_depth_map(image)  # 用局部均值填充缺失处

        image = np.expand_dims(image, 2)        # (H, W) -> (H, W, 1)
        depth_img = image.transpose((2, 0, 1))  # CHW

        return torch.FloatTensor(depth_img), torch.FloatTensor(
            np.squeeze(plane_mask)
        )


class MVTecADRGBDDataset_test(Dataset):
    """测试可视化专用数据集。

    与 ``MVTecADRGBDDataset`` 的区别：
        - ``__getitem__`` 多返回 rgb_path / depth_path / 可视化的 depth 缩略图（uint8），
          方便写图片到磁盘时复原原始路径、展示深度热图。
        - 实例化时会打印 depth_img_process（调试用）。
    """

    def __init__(self, data_dir, transform=None, depth_transform=None, test=False, gt_transform=None, test_min=0.0, test_max=1.0):
        # gt_dir == None --> train；否则 --> test（此注释为原作者遗留说明）
        self.transform = transform
        self.gt_transform = gt_transform
        self.data_info = self.get_data_info(data_dir)
        self.depth_transform = depth_transform

        self.global_max, self.global_min = 1, 0
        if not test:
            for data_info_i in self.data_info:
                rgb_path, depth_path, gt, ad_label, ad_type = data_info_i
                im_min, im_max = get_max_min_depth_img(depth_path)
                self.global_min = min(self.global_min, im_min)
                self.global_max = max(self.global_max, im_max)
            self.global_min = self.global_min * 0.9
            self.global_max = self.global_max * 1.1
        else:
            self.global_max = test_max
            self.global_min = test_min

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, index):
        rgb_path, depth_path, gt, ad_label, ad_type = self.data_info[index]
        rgb_img = Image.open(rgb_path).convert('RGB')
        if self.transform is not None:
            rgb_img = self.transform(rgb_img)

        # 比主类多返回一个 uint8 缩略图 depth_img_process，用于可视化
        depth_img, plane_mask, depth_img_process = self.get_depth_image(depth_path, rgb_img.size()[-2])
        print(depth_img_process)  # 调试打印（可按需移除）

        if self.depth_transform is not None:
            depth_img = torch.cat([depth_img, depth_img, depth_img], dim=0)
            depth_img = self.depth_transform(depth_img)

        if gt == 0:
            gt = torch.zeros([1, rgb_img.size()[-2], rgb_img.size()[-2]])
        else:
            gt = Image.open(gt).convert('L')
            if self.gt_transform is not None:
                gt = self.gt_transform(gt)

        assert rgb_img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return rgb_img, depth_img, gt, ad_label, ad_type, rgb_path, depth_path, depth_img_process

    def get_data_info(self, data_dir):
        """与主类相同，见 ``MVTecADRGBDDataset.get_data_info``。"""
        data_info = list()

        for root, dirs, _ in os.walk(data_dir):
            for sub_dir in dirs:
                rgb_names = os.listdir(os.path.join(root, sub_dir, 'rgb'))
                rgb_names = list(filter(lambda x: x.endswith('.png'), rgb_names))
                for rgb_name in rgb_names:
                    rgb_path = os.path.join(root, sub_dir, 'rgb', rgb_name)
                    depth_name = rgb_name.replace(".png", ".tiff")
                    depth_path = os.path.join(root, sub_dir, 'xyz', depth_name)
                    if sub_dir == 'good':
                        data_info.append((rgb_path, depth_path, 0, 0, sub_dir))
                    else:
                        gt_name = rgb_name
                        gt_path = os.path.join(root, sub_dir, 'gt', gt_name)
                        data_info.append((rgb_path, depth_path, gt_path, 1, sub_dir))

            break

        np.random.shuffle(data_info)

        return data_info

    def get_depth_image(self, file, size=None):
        """与主类相同，另外返回 uint8 可视化缩略图。"""
        depth_img = tifffile.imread(file).astype(np.float32)
        size = self.image_size if size is None else size
        depth_img = cv2.resize(
            depth_img, (size, size), 0, 0, interpolation=cv2.INTER_NEAREST
        )
        depth_img = np.array(depth_img)

        image = depth_img
        image_t = (
            np.array(image)
            .reshape((image.shape[0], image.shape[1], 3))
            .astype(np.float32)
        )
        image = image_t[:, :, 2]

        zero_mask = np.where(image == 0, np.ones_like(image), np.zeros_like(image))
        plane_mask = get_plane_mask(image_t)  # 0 背景, 1 前景
        plane_mask[:, :, 0] = plane_mask[:, :, 0] * (1.0 - zero_mask)
        plane_mask = fill_plane_mask(plane_mask)

        # 历史实验：直接用全局 min/max 归一化（效果不佳，改为逐图归一化）
        # image = fill_depth_map(image)
        # zero_mask = np.where(image == 0, np.ones_like(image), np.zeros_like(image))
        # image = (image - self.global_max) / (self.global_max - self.global_min)
        # image = image * (1.0 - zero_mask)

        image = image * plane_mask[:, :, 0]
        zero_mask = np.where(image == 0, np.ones_like(image), np.zeros_like(image))
        im_min = np.min(image * (1.0 - zero_mask) + 1000 * zero_mask)
        im_max = np.max(image)
        image = (image - im_min) / (im_max - im_min)
        image = image * 0.8 + 0.1
        image = image * (1.0 - zero_mask)
        image = fill_depth_map(image)

        # 可视化缩略图：乘 100 再转 uint8（近似把 [0.1,0.9] 映射到 [10,90]）
        img_process = image * 100
        img_process = img_process.astype(np.uint8)

        image = np.expand_dims(image, 2)
        depth_img = image.transpose((2, 0, 1))

        return torch.FloatTensor(depth_img), torch.FloatTensor(
            np.squeeze(plane_mask)
        ), img_process


