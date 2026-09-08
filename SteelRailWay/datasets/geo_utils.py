# -*- coding: utf-8 -*-
"""
几何预处理工具集合。

本模块为数据集类（MVTec/Eyecandies）提供几何层面的公共函数：
    - 深度图缺失像素填充 ``fill_depth_map``
    - 前景平面分割（RANSAC 风格的三点定平面） ``get_plane_mask`` / ``get_plane_mask_eyecandy``
    - 前景 mask 形态学闭运算 ``fill_plane_mask``
    - Eyecandies 专用的深度图 + pose/intrinsics -> 点云转换 ``depth_to_pointcloud``

所有函数都尽量使用 numpy/opencv，避免不必要的 torch 依赖，便于在数据加载阶段并行调用。
"""

import cv2
import imageio.v3 as iio
import numpy as np
import torch
import yaml


def fill_depth_map(depth_image, iterations=2):
    """用 3x3 邻域的 **有效像素均值** 迭代填充深度图缺失像素。

    实现细节：
        - 采用 torch 的 unfold/fold 高效批量展开 3x3 邻域；
        - 对邻域内非零像素求平均，仅更新原图中为 0 的位置；
        - 迭代 ``iterations`` 次，可以逐步把较大缺失区域由边向内填充。
    """
    dimg = depth_image
    for i in range(iterations):
        zero_mask = np.where(dimg == 0, np.ones_like(dimg), np.zeros_like(dimg))
        dimg_tensor = torch.from_numpy(dimg)
        h, w = dimg_tensor.shape
        dimg_tensor = dimg_tensor.reshape((1, 1, h, w))  # (B, C, H, W)
        # 把每个 3x3 邻域展平为长度 9 的向量
        dimg_t = torch.nn.functional.unfold(
            dimg_tensor, 3, dilation=1, padding=1, stride=1
        )  # B, 1*3*3, L=H*W
        # 统计邻域内有效（非 0）像素数量和其总和，算均值
        dimg_t_nonzero_sum = torch.sum(
            torch.where(dimg_t > 0, torch.ones_like(dimg_t), torch.zeros_like(dimg_t)),
            dim=1,
            keepdim=True,
        )
        dimg_t_sum = torch.sum(dimg_t, dim=1, keepdim=True)
        dimg_t_filtered = dimg_t_sum / (dimg_t_nonzero_sum + 1e-12)
        # 折回原始 HxW 形状
        dimg_out = torch.nn.functional.fold(
            dimg_t_filtered, dimg.shape[:2], 1, dilation=1, padding=0, stride=1
        )  # (B, 1, H, W)
        # 只用邻域均值替换原来为 0 的像素，其它位置保持不变
        dimg = dimg_out.numpy()[0, 0, :, :] * zero_mask + (1.0 - zero_mask) * dimg
    return dimg


def fill_plane_mask(plane_mask):
    """对前景 mask 做形态学闭运算（先膨胀后腐蚀），平滑边缘、填补小洞。"""
    kernel = np.ones((3, 3), np.uint8)
    plane_mask[:, :, 0] = cv2.morphologyEx(
        plane_mask, cv2.MORPH_CLOSE, kernel, iterations=3
    )
    return plane_mask


def get_plane_mask(depth_image):
    """基于 3 个角点拟合平面、再用点到平面距离阈值分割前景。

    - 分别取左上、右上、左下三个 3x3 角块的"有效像素均值"作为 3 个平面点。
    - 用三点定平面 ``get_plane_from_points`` 得到 ax+by+cz=d。
    - 距离平面 > 0.005 米（阈值经验值）视为前景（物体），=1 标记。
    """
    h, w, c = depth_image.shape
    points = np.reshape(depth_image, (h * w, c))

    # 用 max(..., 1e-12) 防止角块全为 0 时除零
    p1 = np.sum(depth_image[:3, :3, :], axis=(0, 1)) / max(np.sum(depth_image[:3, :3, 2] != 0), 1e-12)
    p2 = np.sum(depth_image[:3, -3:, :], axis=(0, 1)) / max(np.sum(depth_image[:3, -3:, 2] != 0), 1e-12)
    p3 = np.sum(depth_image[-3:, :3, :], axis=(0, 1)) / max(np.sum(depth_image[-3:, :3, 2] != 0), 1e-12)

    plane = get_plane_from_points(p1, p2, p3)
    point_distance = get_distance_to_plane(points, np.array(plane))
    # 距离阈值 0.005：MVTec 3D 场景经验值，超出此距离视为物体
    points_mask = np.where(
        point_distance > 0.005,
        np.ones_like(point_distance),
        np.zeros_like(point_distance),
    )
    plane_mask = np.reshape(points_mask, (h, w, 1))
    return plane_mask


def get_plane_from_points(p1, p2, p3):
    """三点定平面：计算 ax+by+cz=d 的系数。"""
    # 两个在平面内的向量
    v1 = p3 - p1
    v2 = p2 - p1

    # 叉积得到平面法向量 (a, b, c)
    cp = np.cross(v1, v2)
    a, b, c = cp

    # d = (a,b,c) · p3，即平面方程常数项
    d = np.dot(cp, p3)
    return a, b, c, d


def get_distance_to_plane(points, plane):
    """批量计算点到平面的距离。

    参数：
        points : (N, 3) 三维点坐标
        plane  : (a, b, c, d) 平面方程
    返回：
        (N,) 每个点到平面的距离（绝对值）
    """
    plane_rs = np.expand_dims(plane, 0)
    dist = (
        np.abs(np.sum(points * plane_rs[:, :-1], axis=1) - plane[-1])
        / np.sum(plane[:-1] ** 2) ** 0.5
    )
    return dist


def get_plane_points(p, plane):
    """把任意点 p 投影到给定平面上的最近点。

    注：此函数当前未在主流程中使用，保留作后续可视化/分析。
    """
    a, b, c, d = plane
    normal = np.zeros(1, 3)
    normal[0, :] = np.array([a, b, c])
    # 沿法向量移动一段距离使点落在平面上
    c = (d - np.sum(p * normal, axis=1)) / np.sum(normal**2)
    out_points = p + c * normal
    return out_points


def get_distance_to_plane_eyecandy(points, plane):
    """Eyecandies 专用的稳健版本：法向量模长为 0 时返回 0 而非 NaN。"""
    plane_rs = np.expand_dims(plane, 0)
    dist = (
        np.abs(np.sum(points * plane_rs[:, :-1], axis=1) - plane[-1])
        / (np.sum(plane[:-1] ** 2) ** 0.5 + 1e-12)
    )
    return dist


def get_plane_mask_eyecandy(depth_image, thr=0.005):
    """Eyecandies 专用的前景 mask：三点取图像边缘中点和两个下角点。

    不同于 MVTec 使用三个角落，这里使用 (上中, 左下, 右下) 三个点，
    更契合 Eyecandies 的相机视角和场景布局。
    """
    h, w, c = depth_image.shape
    points = np.reshape(depth_image, (h * w, c))
    p1 = depth_image[0, w // 2, :]       # 顶部中点
    p2 = depth_image[h - 1, 0, :]        # 左下角
    p3 = depth_image[h - 1, w - 1, :]    # 右下角
    plane = get_plane_from_points(p1, p2, p3)
    point_distance = get_distance_to_plane_eyecandy(points, np.array(plane))
    points_mask = np.where(
        point_distance > thr,
        np.ones_like(point_distance),
        np.zeros_like(point_distance),
    )
    plane_mask = np.reshape(points_mask, (h, w, 1))
    return plane_mask


def load_and_convert_depth(depth_img, info_depth):
    """读取 Eyecandies 的 16bit 深度 PNG 并转换回米单位。

    info_depth 是 yaml 文件，包含 ``normalization.{min, max}``。
    16bit 图的 [0, 65535] 被映射回 [min, max] 米。
    """
    with open(info_depth) as f:
        data = yaml.safe_load(f)
    mind, maxd = data["normalization"]["min"], data["normalization"]["max"]
    dimg = iio.imread(depth_img)
    dimg = dimg.astype(np.float32)
    dimg = dimg / 65535.0 * (maxd - mind) + mind
    return dimg


def depth_to_pointcloud(depth_img, info_depth, pose_txt):
    """把深度图 + 相机 pose + 内参转换为世界坐标系下的点云 (H, W, 3)。

    步骤：
        1. 读取米单位深度图；读取 4x4 相机 pose 矩阵。
        2. 构造 3x3 相机内参（这里焦距固定 711.11，主点取图像中心）。
        3. ``camera_proj = K @ pose``，然后对每个像素生成 (u, v, 1, 1/depth)。
        4. 用 ``camera_proj`` 的逆把像素反投影到世界坐标系。
        5. 乘以深度值即得到 3D 点。
    """
    # 输入深度图（米制），下面内参是 Eyecandies 默认值
    focal_length = 711.11
    depth_mt = load_and_convert_depth(depth_img, info_depth)
    # 相机 pose（4x4 矩阵）
    pose = np.loadtxt(pose_txt)
    # 构造 4x4 内参矩阵（主点位于图像中心）
    height, width = depth_mt.shape[:2]
    intrinsics_4x4 = np.array(
        [
            [focal_length, 0, width / 2, 0],
            [0, focal_length, height / 2, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    # 相机投影矩阵 = 内参 @ 位姿
    camera_proj = intrinsics_4x4 @ pose
    # 为每个像素构造 (u, v, 1, 1/depth) 齐次坐标向量
    coords_x = np.expand_dims(
        np.tile(np.arange(0, width), (height, 1)), axis=2
    )  # (H, W, 1)
    coords_y = np.expand_dims(
        np.tile(np.arange(0, height), (width, 1)).T, axis=2
    )  # (H, W, 1)
    ones_t = np.ones((height, width, 1))
    depth_inv = np.expand_dims(1.0 / depth_mt, axis=2)
    camera_vectors = np.concatenate(
        (coords_x, coords_y, ones_t, depth_inv), axis=2
    )  # (H, W, 4)
    # 通过 camera_proj 的逆做反投影：(4,4) x (4,H,W) -> (4,H,W)
    hom_3d_pts = np.einsum(
        "ij,jlm->ilm", np.linalg.inv(camera_proj), camera_vectors.transpose((2, 0, 1))
    )
    # 乘以深度得到真实 3D 坐标（单位：米）
    pcd = depth_mt.reshape(height, width, 1) * hom_3d_pts.transpose((1, 2, 0))  # (H, W, 3)
    return pcd
