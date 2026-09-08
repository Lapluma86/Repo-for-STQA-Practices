"""
通用评估工具函数（来自 MVTec 3D-AD 官方评估脚本的精简版）。

主要包含：
    - ``OBJECT_NAMES`` : MVTec 3D-AD 的 10 个物体类别名称（遍历数据集时用）。
    - ``trapezoid``    : 带可选上界 x_max 的梯形法数值积分，
                         常用于在指定 FPR 范围内计算 PRO-AUC。
    - ``generate_toy_dataset`` : 生成随机 anomaly map 和带矩形 gt 的小数据集，
                                 用于自测评估流程是否正确。
"""
from bisect import bisect

import numpy as np

# MVTec 3D-AD 基准的 10 个类别（顺序与官方一致）
OBJECT_NAMES = ['bagel', 'cable_gland', 'carrot', 'cookie', 'dowel', 'foam',
                'peach', 'potato', 'rope', 'tire']


def trapezoid(x, y, x_max=None):
    """带可选上界的梯形法数值积分。

    相较 ``numpy.trapz``，本函数允许指定积分上界 ``x_max``：
    当 x_max 不在 x 采样点中时，会在 x_max 所在区间内线性插值出 y 值，
    然后把这段"零头"梯形面积加进去。

    参数：
        x     : 升序采样点（允许重复，但重复点的 y 顺序会影响积分）
        y     : 对应的函数值
        x_max : 积分上限；None 表示用 x 的最大值
    返回：
        曲线下面积（标量）。

    说明：非有限值（NaN/Inf）会被丢弃并打印警告。
    """

    x = np.asarray(x)
    y = np.asarray(y)
    # 过滤 NaN/Inf
    finite_mask = np.logical_and(np.isfinite(x), np.isfinite(y))
    if not finite_mask.all():
        print("""WARNING: Not all x and y values passed to trapezoid(...)
                 are finite. Will continue with only the finite values.""")
    x = x[finite_mask]
    y = y[finite_mask]

    # 计算 x_max 不在采样点时的额外修正项
    correction = 0.
    if x_max is not None:
        if x_max not in x:
            # 找到插入 x_max 后仍保持排序的位置
            ins = bisect(x, x_max)
            # x_max 必须落在采样区间内（不能在两端）
            assert 0 < ins < len(x)

            # 在 (x[ins-1], x[ins]) 之间线性插值得到 y_interp，
            # 然后把 x[ins-1] -> x_max 这一小段的梯形面积加进 correction
            y_interp = y[ins - 1] + ((y[ins] - y[ins - 1]) *
                                     (x_max - x[ins - 1]) /
                                     (x[ins] - x[ins - 1]))
            correction = 0.5 * (y_interp + y[ins - 1]) * (x_max - x[ins - 1])

        # 截断到 x_max 之前的全部采样点
        mask = x <= x_max
        x = x[mask]
        y = y[mask]

    # 梯形法：sum(0.5 * (y_i + y_{i+1}) * (x_{i+1} - x_i)) + 修正项
    return np.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])) + correction


def generate_toy_dataset(num_images, image_width, image_height, gt_size):
    """生成一个 toy 数据集用来单元测试评估流程。

    - 每张图的 anomaly map 是随机噪声
    - ground truth 都是左上角 gt_size × gt_size 的矩形区域为异常

    返回：
        anomaly_maps     : list of (H, W) 随机数组
        ground_truth_maps: list of (H, W) 0/1 数组
    """
    # 固定随机种子，保证可复现
    np.random.seed(1338)

    anomaly_maps = []
    ground_truth_maps = []
    for _ in range(num_images):
        # 随机异常热图
        anomaly_map = np.random.random((image_height, image_width))

        # 固定位置的矩形 gt
        ground_truth_map = np.zeros((image_height, image_width))
        ground_truth_map[0:gt_size, 0:gt_size] = 1

        anomaly_maps.append(anomaly_map)
        ground_truth_maps.append(ground_truth_map)

    return anomaly_maps, ground_truth_maps
