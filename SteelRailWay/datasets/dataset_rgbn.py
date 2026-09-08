# -*- coding: utf-8 -*-
"""
MVTec 3D-AD 数据集 RGB + Normal（RGBN）加载器。

与 RGBD 版本（``dataset_rgbd.py``）的差别：辅助模态不是原始深度图，
而是由深度图经 Sobel 梯度计算得到的 **法向量图**。法向量对平坦/细节变化
更敏感，在小尺度表面异常检测中通常比深度更有信息量。

流程：
    1. 读取 xyz 深度图；估计前景平面 mask；对深度做逐图归一化与缺失填充。
    2. 用 ``depth_to_normal_map`` 基于 Sobel 计算法向量图，归一到 [0, 1]。
    3. 用前景 mask 屏蔽背景区域（只保留物体的法向信息）。
    4. resize 到目标分辨率（法向用双线性，mask 用最近邻）。
    5. 同时支持 normal 结果缓存，避免训练期重复计算。
"""

from torch.utils.data import Dataset
import os
from PIL import Image

from .geo_utils import *  # 提供 np / cv2 / torch / get_plane_mask / fill_* 等工具

import tifffile


def get_max_min_depth_img(image_path):
    """读取单张 xyz 深度图，返回其 z 通道的有效最小/最大值（排除 0 缺失点）。"""
    image = tifffile.imread(image_path).astype(np.float32)
    image_t = (
        np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32)
    )
    image = image_t[:, :, 2]
    zero_mask = np.where(image == 0, np.ones_like(image), np.zeros_like(image))
    im_max = np.max(image)
    im_min = np.min(image * (1.0 - zero_mask) + 1000 * zero_mask)
    return im_min, im_max


def depth_to_normal_map(depth, k=5, mask=None):
    """由 2D 深度图计算近似法向量图。

    原理：把 (x, y, z) 曲面的局部法向量近似为 (-dz/dx, -dz/dy, 1)，
    然后单位化得到方向向量，再从 [-1, 1] 线性映射到 [0, 1] 以便作为图像使用。

    参数：
        depth : (H, W) float，归一化后的深度图
        k     : Sobel 滤波器大小，越大平滑越强（抑制噪声但损失细节）
        mask  : 可选的 (H, W) 前景 mask，背景处会置零
    返回：
        (H, W, 3) 的 float 法向量"伪彩图"，每个通道 ∈ [0, 1]
    """
    # x/y 方向的深度梯度
    dzdx = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=k)
    dzdy = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=k)

    # 用 (-dzdx, -dzdy, 1) 作为法向量，再做单位化
    normal = np.dstack((-dzdx, -dzdy, np.ones_like(depth)))
    norm = np.linalg.norm(normal, axis=2, keepdims=True)
    normal /= (norm + 1e-8)

    # 把方向从 [-1, 1] 映射到 [0, 1]（图像范围）
    normal_img = (normal + 1) / 2.0

    if mask is not None:
        normal_img *= mask[:, :, None]  # 背景置零

    return normal_img


class MVTecADRGBNDataset(Dataset):
    """RGB + Normal 数据集。

    新增相对 RGBD 版本的能力：
        - ``cache_normals`` : 在训练阶段把计算过的 (normal, mask) 缓存到内存，
          避免每个 epoch 反复读 tiff + 做 Sobel 卷积（这两步较昂贵）。
        - ``k`` : Sobel 核大小，可调节法向图的平滑度。
    """

    def __init__(self, data_dir, transform=None, depth_transform=None, test=False, gt_transform=None, test_min=0.0, test_max=1.0, k=5, cache_normals=True):
        self.transform = transform
        self.gt_transform = gt_transform
        self.data_info = self.get_data_info(data_dir)
        self.depth_transform = depth_transform

        # 预处理缓存：depth->normal 的计算很重（tiff 读取、mask、Sobel），默认对每个样本缓存一次
        # 测试阶段禁用缓存（避免数据增强随机性导致错误复用）
        self.cache_normals = cache_normals and (not test)
        self._normal_cache = {}  # depth_path -> (normal_img_tensor, plane_mask_tensor)

        # 计算/使用全局深度范围（逻辑与 RGBD 版本一致）
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

        self.k = k  # Sobel 卷积核大小

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, index):
        rgb_path, depth_path, gt, ad_label, ad_type = self.data_info[index]

        # RGB 读取与 transform
        rgb_img = Image.open(rgb_path).convert('RGB')
        if self.transform is not None:
            rgb_img = self.transform(rgb_img)

        # 法向图：优先读缓存，否则现算并入缓存
        if self.cache_normals and depth_path in self._normal_cache:
            normal_img, plane_mask = self._normal_cache[depth_path]
        else:
            normal_img, plane_mask = self.get_normal_image(depth_path, rgb_img.size()[-2], self.k)
            if self.cache_normals:
                self._normal_cache[depth_path] = (normal_img, plane_mask)

        # 对法向图应用与 RGB 相同的标准化 transform（归一化均值/方差）
        if self.depth_transform is not None:
            normal_img = self.depth_transform(normal_img)

        # 正常样本无 gt，用全零 tensor 占位
        if gt == 0:
            gt = torch.zeros([1, rgb_img.size()[-2], rgb_img.size()[-2]])
        else:
            gt = Image.open(gt).convert('L')
            if self.gt_transform is not None:
                gt = self.gt_transform(gt)

        return rgb_img, normal_img, gt, ad_label, ad_type

    def get_data_info(self, data_dir):
        """扫描 MVTec 3D-AD 目录结构，收集 (rgb, xyz, gt_or_0, label, type) 列表。

        结构：data_dir/<type>/{rgb, xyz, gt}/*
        "good" 子目录为正常样本。
        """
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

            break  # 只取第一层子目录

        np.random.shuffle(data_info)

        return data_info

    def get_normal_image(self, file, target_size=None, k=5):
        """读取 xyz 图 → 深度归一化 + 填充 → Sobel 计算法向量 → resize。

        返回：
            normal_img  : (3, H, W) 的 FloatTensor，值域 [0, 1]
            plane_mask  : (H, W) 的 FloatTensor，1=前景，0=背景
        """
        # 读 (H, W, 3) 的 xyz，保留原分辨率做 normal（避免插值后梯度失真）
        xyz_data = tifffile.imread(file).astype(np.float32)
        H_orig, W_orig, C = xyz_data.shape

        size = self.image_size if target_size is None else target_size

        # 取 z 通道作为深度
        original_depth = xyz_data[:, :, 2]
        image = original_depth
        image_t = xyz_data

        # 前景 mask：排除零值缺失 + 平面拟合去掉背景
        zero_mask = np.where(image == 0, np.ones_like(image), np.zeros_like(image))
        plane_mask = get_plane_mask(image_t)  # 0 背景, 1 前景
        plane_mask[:, :, 0] = plane_mask[:, :, 0] * (1.0 - zero_mask)
        plane_mask = fill_plane_mask(plane_mask)
        plane_mask_2d = plane_mask[:, :, 0]

        # 只在前景区域做逐图 min-max 归一化
        image = image * plane_mask[:, :, 0]
        zero_mask = np.where(image == 0, np.ones_like(image), np.zeros_like(image))
        im_min = np.min(image * (1.0 - zero_mask) + 1000 * zero_mask)
        im_max = np.max(image)
        image = (image - im_min) / (im_max - im_min)
        # image = image * 0.8 + 0.1  # RGBN 版本直接使用 [0,1]，不压缩到 [0.1,0.9]
        image = image * (1.0 - zero_mask)  # 缺失保持 0
        image = fill_depth_map(image)       # 用局部均值填补缺失

        filled_normalized_depth = image

        # Sobel 近似法向量（在原分辨率上计算，保留高频细节）
        normals_map_raw = depth_to_normal_map(filled_normalized_depth, k=k)  # HxWx3

        normals_map_processed = normals_map_raw

        # 屏蔽背景区域（法向量只对前景有意义）
        plane_mask_3d = np.expand_dims(plane_mask_2d, axis=2)
        final_normals = normals_map_processed * plane_mask_3d

        # 缩放到网络期望的大小；法向用线性插值以保持方向平滑
        final_normals_resized = cv2.resize(
            final_normals, (size, size),
            interpolation=cv2.INTER_LINEAR
        )

        normal_img = final_normals_resized.transpose((2, 0, 1))  # HWC -> CHW

        # mask 用最近邻插值避免边缘模糊
        plane_mask_resized = cv2.resize(
            plane_mask[:, :, 0], (size, size),
            interpolation=cv2.INTER_NEAREST
        )
        plane_mask_resized = np.expand_dims(plane_mask_resized, 2)

        return torch.FloatTensor(normal_img), torch.FloatTensor(
            np.squeeze(plane_mask_resized)
        )
