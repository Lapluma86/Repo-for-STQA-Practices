"""
PRO（Per-Region Overlap）曲线与 AU-PRO 计算工具。

对比 ``metrics_utils.py``：本文件采用的是更高效的 **全局排序 + 累积计数** 实现，
一次性对所有像素做 argsort，然后用 cumsum 同时增量更新 FPR 和 PRO，
复杂度 O(N log N)，适合像素总数较大的数据集。

流程示意：
    1. 为每张 gt 图计算连通分量：得到 "OK pixels 掩码" 和 "每个异常像素属于哪个区域"。
    2. 构造两个与像素一一对应的"变化量"数组：
        - fp_change  : OK 像素处 +1，代表该像素被当作"预测为异常"时假正率增加量（未归一化）。
        - pro_change : 异常像素处为 1/region_size，代表该像素被检出时该区域的 overlap 增加多少。
    3. 按异常分数降序把所有像素排成一列，累加 fp_change / pro_change，
       再除以总正/总区域数即得到 FPR / PRO 曲线。
    4. 合并同分数的连续点（只保留每个阈值的最后一个），并裁剪到 [0, 1]。
"""
import numpy as np
from scipy.ndimage.measurements import label
from .generic_util import trapezoid


def compute_pro_util(anomaly_maps, ground_truth_maps):
    """高效计算 (FPR, PRO) 曲线。

    参数：
        anomaly_maps      : list of (H, W) float anomaly map
        ground_truth_maps : list of (H, W) 0/1 gt mask
    返回：
        fprs : 1D numpy，升序排列
        pros : 1D numpy，与 fprs 一一对应
    """

    # 3x3 结构元：8 邻域做连通分量
    structure = np.ones((3, 3), dtype=int)

    num_ok_pixels = 0    # 整个数据集的 OK 像素总数
    num_gt_regions = 0   # 整个数据集的 gt 连通分量总数

    # 预分配"每个像素贡献的 FPR/PRO 增量"数组，形状与输入一致
    shape = (len(anomaly_maps),
             anomaly_maps[0].shape[0],
             anomaly_maps[0].shape[1])
    fp_changes = np.zeros(shape, dtype=np.uint32)
    # 数据量过大时 uint32 可能溢出，这里断言提醒改 uint64
    assert shape[0] * shape[1] * shape[2] < np.iinfo(fp_changes.dtype).max, \
        'Potential overflow when using np.cumsum(), consider using np.uint64.'

    pro_changes = np.zeros(shape, dtype=np.float64)

    for gt_ind, gt_map in enumerate(ground_truth_maps):
        # 连通分量：labeled==0 为背景 OK，其它为各异常区域
        labeled, n_components = label(gt_map, structure)
        num_gt_regions += n_components

        ok_mask = labeled == 0
        num_ok_pixels_in_map = np.sum(ok_mask)
        num_ok_pixels += num_ok_pixels_in_map

        # OK 像素的 fp_change = 1（被误判时 FPR 分子 +1）
        fp_change = np.zeros_like(gt_map, dtype=fp_changes.dtype)
        fp_change[ok_mask] = 1

        # 异常像素的 pro_change = 1 / region_size
        # （该像素被正确检出时，对应区域的 overlap 增加 1/region_size）
        pro_change = np.zeros_like(gt_map, dtype=np.float64)
        for k in range(n_components):
            region_mask = labeled == (k + 1)
            region_size = np.sum(region_mask)
            pro_change[region_mask] = 1. / region_size

        fp_changes[gt_ind, :, :] = fp_change
        pro_changes[gt_ind, :, :] = pro_change

    # 把所有像素按分数一维化后一起处理
    anomaly_scores_flat = np.array(anomaly_maps).ravel()
    fp_changes_flat = fp_changes.ravel()
    pro_changes_flat = pro_changes.ravel()

    # 按异常分数 **降序** 排序（阈值从高到低扫描）
    sort_idxs = np.argsort(anomaly_scores_flat).astype(np.uint32)[::-1]

    # 原地 take 比 fancy indexing 更省内存
    np.take(anomaly_scores_flat, sort_idxs, out=anomaly_scores_flat)
    anomaly_scores_sorted = anomaly_scores_flat
    np.take(fp_changes_flat, sort_idxs, out=fp_changes_flat)
    fp_changes_sorted = fp_changes_flat
    np.take(pro_changes_flat, sort_idxs, out=pro_changes_flat)
    pro_changes_sorted = pro_changes_flat

    del sort_idxs

    # 用 cumsum 得到每个阈值对应的 (FPR, PRO)：阈值越低，预测为异常的像素越多
    np.cumsum(fp_changes_sorted, out=fp_changes_sorted)
    fp_changes_sorted = fp_changes_sorted.astype(np.float32, copy=False)
    np.divide(fp_changes_sorted, num_ok_pixels, out=fp_changes_sorted)
    fprs = fp_changes_sorted

    np.cumsum(pro_changes_sorted, out=pro_changes_sorted)
    np.divide(pro_changes_sorted, num_gt_regions, out=pro_changes_sorted)
    pros = pro_changes_sorted

    # 合并相同分数的连续点：只保留最后一个（因为它累积了该阈值全部变化）
    # 例：sorted_scores = [7, 4, 4, 4, 3, 1, 1] => keep_mask = [T, F, F, T, T, F, T]
    keep_mask = np.append(np.diff(anomaly_scores_sorted) != 0, np.True_)
    del anomaly_scores_sorted

    fprs = fprs[keep_mask]
    pros = pros[keep_mask]
    del keep_mask

    # 防止浮点累积误差导致 > 1，裁剪到 [0, 1]
    np.clip(fprs, a_min=None, a_max=1., out=fprs)
    np.clip(pros, a_min=None, a_max=1., out=pros)

    # 在曲线两端追加 (0, 0) 和 (1, 1) 便于后续做面积积分
    zero = np.array([0.])
    one = np.array([1.])

    return np.concatenate((zero, fprs, one)), np.concatenate((zero, pros, one))


def compute_pro(anomaly_maps, ground_truth_maps, integration_limit=0.3):
    """综合 API：得到 PRO 曲线后用梯形法求 AU-PRO（默认截取 FPR ≤ 0.3）。"""
    all_fprs, all_pros = compute_pro_util(
        anomaly_maps=anomaly_maps,
        ground_truth_maps=ground_truth_maps)

    au_pro = trapezoid(all_fprs, all_pros, x_max=integration_limit)
    au_pro /= integration_limit  # 归一化到 [0, 1]

    return au_pro


def main():
    """本文件单独运行时的自测：用 toy 数据跑一遍 AU-PRO。"""

    from generic_util import trapezoid, generate_toy_dataset

    integration_limit = 0.3

    # 生成 toy 数据
    anomaly_maps, ground_truth_maps = generate_toy_dataset(
        num_images=200, image_width=500, image_height=300, gt_size=10)

    # 计算 PRO 曲线并积分
    all_fprs, all_pros = compute_pro_util(
        anomaly_maps=anomaly_maps,
        ground_truth_maps=ground_truth_maps)

    au_pro = trapezoid(all_fprs, all_pros, x_max=integration_limit)
    au_pro /= integration_limit
    print(f"AU-PRO (FPR limit: {integration_limit}): {au_pro}")


if __name__ == "__main__":
    main()
