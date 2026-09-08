"""
MVTec 3D-AD 官方评估脚本的精简版：PRO/AUC 指标计算工具。

基于 https://www.mydrive.ch/shares/45924/.../evaluation_code.tar.xz
主要对外 API：
    - ``calculate_au_pro``  : 给定 gt 列表与预测异常图列表，
                              返回多个 FPR 上限下的 PRO-AUC（默认 [0.3, 0.1, 0.05, 0.01]）。
    - ``calculate_au_prc``  : 基于像素级 ROC 的 AUC（注意变量命名 PRC 但实际是 ROC-AUC）。

关键优化：相较 ``eval_utils.compute_pro`` 里"阈值循环 + 对每张图 regionprops"的 O(T·N) 朴素实现，
本文件使用 ``GroundTruthComponent`` 预先保存每个连通分量像素的"异常分数排序数组"+指针，
从而把不同阈值下的 overlap 计算降至 O(T + N) 摊销，大数据集速度差异显著。
"""
import numpy as np
import sklearn
from scipy.ndimage.measurements import label
from bisect import bisect


class GroundTruthComponent:
    """单个 gt 连通分量对应的异常分数容器。

    优化技巧：
        - 把分量内所有像素的异常分数排序一次；
        - 维护一个指针 ``index``，代表"小于等于当前阈值"的像素数；
        - 要求阈值按递增顺序传入（``compute_overlap``），这样指针只需单调右移；
        - 于是在多阈值扫描中每个分量只遍历一次自己的像素集合。
    """

    def __init__(self, anomaly_scores):
        """
        参数：
            anomaly_scores : 该 gt 连通区域内所有像素的异常分数数组（1D）
        """
        # 排序一份分数（拷贝避免原数组被修改）
        self.anomaly_scores = anomaly_scores.copy()
        self.anomaly_scores.sort()

        # 指针：当前阈值下"被判定为非异常"的像素数
        self.index = 0

        # 上一次被评估的阈值（用于断言阈值是递增的）
        self.last_threshold = None

    def compute_overlap(self, threshold):
        """返回当前阈值下该 gt 区域被正确标记为异常的像素比例。

        要求：多次调用时 threshold 必须非降。
        """
        if self.last_threshold is not None:
            assert self.last_threshold <= threshold

        # 指针向右扫过所有 <= threshold 的分数（这些像素会被判定为 OK，即漏检）
        while (self.index < len(self.anomaly_scores) and self.anomaly_scores[self.index] <= threshold):
            self.index += 1

        # 剩余像素才是被正确检出的异常像素，占比就是 overlap
        return 1.0 - self.index / len(self.anomaly_scores)


def trapezoid(x, y, x_max=None):
    """带可选上界 x_max 的梯形法积分。

    逻辑与 ``generic_util.trapezoid`` 一致，此处为本文件内部的拷贝以避免模块间互相依赖。
    参数与返回值见 ``generic_util.trapezoid``。
    """

    x = np.array(x)
    y = np.array(y)
    finite_mask = np.logical_and(np.isfinite(x), np.isfinite(y))
    if not finite_mask.all():
        print(
            """WARNING: Not all x and y values passed to trapezoid are finite. Will continue with only the finite values.""")
    x = x[finite_mask]
    y = y[finite_mask]

    # x_max 不在采样点时的修正项（插值补齐最后一段梯形）
    correction = 0.
    if x_max is not None:
        if x_max not in x:
            ins = bisect(x, x_max)
            assert 0 < ins < len(x)

            y_interp = y[ins - 1] + ((y[ins] - y[ins - 1]) * (x_max - x[ins - 1]) / (x[ins] - x[ins - 1]))
            correction = 0.5 * (y_interp + y[ins - 1]) * (x_max - x[ins - 1])

        mask = x <= x_max
        x = x[mask]
        y = y[mask]

    return np.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])) + correction


def collect_anomaly_scores(anomaly_maps, ground_truth_maps):
    """从 (anomaly_map, gt_map) 对中提取评估所需的两类分数集合。

    返回：
        ground_truth_components  : GroundTruthComponent 列表，每个对应一个 gt 连通分量
        anomaly_scores_ok_pixels : 所有"正常"像素的异常分数（升序排序）
                                   用于快速按位置定位到指定 FPR 对应的阈值
    """
    assert len(anomaly_maps) == len(ground_truth_maps)

    # 预分配所有像素总数的一维数组（放 OK 像素的分数）
    ground_truth_components = []
    anomaly_scores_ok_pixels = np.zeros(len(ground_truth_maps) * ground_truth_maps[0].size)

    # 3x3 结构元：8-邻域连通分量
    structure = np.ones((3, 3), dtype=int)

    ok_index = 0
    for gt_map, prediction in zip(ground_truth_maps, anomaly_maps):

        # 连通分量标签：labeled==0 是背景，1..n 是各异常区域
        labeled, n_components = label(gt_map, structure)

        # 背景像素的异常分数 -> 候选假正率池
        num_ok_pixels = len(prediction[labeled == 0])
        anomaly_scores_ok_pixels[ok_index:ok_index + num_ok_pixels] = prediction[labeled == 0].copy()
        ok_index += num_ok_pixels

        # 每个 gt 连通分量构造一个 GroundTruthComponent
        for k in range(n_components):
            component_scores = prediction[labeled == (k + 1)]
            ground_truth_components.append(GroundTruthComponent(component_scores))

    # 截断未使用的尾部并排序，便于后续按位置定阈值
    anomaly_scores_ok_pixels = np.resize(anomaly_scores_ok_pixels, ok_index)
    anomaly_scores_ok_pixels.sort()

    return ground_truth_components, anomaly_scores_ok_pixels


def compute_pro(anomaly_maps, ground_truth_maps, num_thresholds):
    """在等距阈值上计算 PRO 曲线（FPR, PRO）。

    策略：
        - 先把所有 OK 像素的异常分数排序；
        - 在这个有序数组上等距采样 num_thresholds 个位置作为"阈值候选"，
          这样能保证 FPR 近似均匀分布，而不是简单的"在分数区间内均匀"；
        - 对每个阈值更新所有 GT 分量的 overlap 取平均得到 PRO，同时从位置直接算出 FPR。
    返回：
        fprs : 按 FPR 升序排列的假正率列表
        pros : 对应的 PRO 值列表
        thr  : 对应的阈值列表（用于调试/可视化）
    """
    ground_truth_components, anomaly_scores_ok_pixels = collect_anomaly_scores(anomaly_maps, ground_truth_maps)

    # 在 OK 像素数量内等距选 num_thresholds 个位置
    threshold_positions = np.linspace(0, len(anomaly_scores_ok_pixels) - 1, num=num_thresholds, dtype=int)

    # 起点 (FPR, PRO) = (1, 1)：阈值无限低时全部预测为异常
    fprs = [1.0]
    pros = [1.0]
    thr = [0.0]

    for pos in threshold_positions:
        threshold = anomaly_scores_ok_pixels[pos]

        # 用排序数组的位置直接得到 FPR：位置越靠后意味着阈值越高、FP 越少
        fpr = 1.0 - (pos + 1) / len(anomaly_scores_ok_pixels)

        # PRO = 所有连通分量 overlap 的算术平均
        pro = 0.0
        for component in ground_truth_components:
            pro += component.compute_overlap(threshold)
        pro /= len(ground_truth_components)

        fprs.append(fpr)
        pros.append(pro)
        thr.append(threshold)

    # 反转列表使 FPR 按升序
    fprs = fprs[::-1]
    pros = pros[::-1]
    thr = thr[::-1]

    return fprs, pros, thr


def calculate_au_pro(gts, predictions, integration_limit = [0.3, 0.1, 0.05, 0.01], num_thresholds = 100):
    """给定某类物体的所有 gt 和预测图，计算多个 FPR 上限下的 PRO-AUC。

    参数：
        gts               : gt 列表（每项 2D numpy，取值 {0,1}）
        predictions       : 对应异常热图列表
        integration_limit : 积分上界 FPR 列表，每项都会输出一个 AUC
        num_thresholds    : PRO 曲线采样点数
    返回：
        au_pros   : 对应每个 integration_limit 的 PRO-AUC（已除以上限归一化到 [0,1]）
        pro_curve : (fprs, pros, thrs) 三元组，供画图/调试使用
    """
    pro_curve = compute_pro(anomaly_maps = predictions, ground_truth_maps = gts, num_thresholds = num_thresholds)

    au_pros = []
    for int_lim in integration_limit:
        # 在指定 FPR 上界内做梯形积分，然后除以上限使结果在 [0, 1]
        au_pro = trapezoid(pro_curve[0], pro_curve[1], x_max = int_lim)
        au_pro /= int_lim
        au_pros.append(au_pro)

    return au_pros, pro_curve


def calculate_au_prc(gts, predictions):
    """像素级 ROC-AUC（函数名延续原作者"PRC"但内部实际使用 ROC 曲线）。

    直接把所有像素的 gt/score 扁平化后交给 sklearn.metrics.roc_curve + auc。
    """
    fpr, tpr, _ = sklearn.metrics.roc_curve(gts, predictions)
    au_prc = sklearn.metrics.auc(fpr, tpr)

    return au_prc