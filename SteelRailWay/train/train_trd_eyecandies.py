# -*- coding: utf-8 -*-
"""
Eyecandies 数据集上的 TRD（Teacher-Reverse-Distillation）双模态训练/评估入口。

架构概览（RGB + Normals 双支）：
    - 教师 teacher_rgb / teacher_depth : Wide-ResNet50_2 编码器（冻结、eval 模式）。
    - 学生 student_rgb / student_depth : ``ResNet50DualModalDecoder``。
        * student_rgb  以 teacher_rgb   的特征为主、teacher_depth 的特征为辅；
        * student_depth 反过来。
    - 两个学生并行训练，彼此独立优化器。
    - 训练目标（``loss_distil``）把主模态教师特征作为对齐目标，并对 Projector
      的中间输出（proj_d / proj_d_amply / output_Sr_am）都做余弦蒸馏。

测试：
    - 用 ``cal_anomaly_map`` 的 add 模式分别得到 RGB / Normal 分支的异常图；
    - 用验证集统计各分支均值/方差做 z-score 归一化，再相加得到最终异常图；
    - 最后用高斯滤波平滑；支持三种指标视角：融合、纯 RGB、纯 Normal。
"""

import torch
# >>> path-bootstrap (auto-added on restructure) >>>
import os, sys
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)
# <<< path-bootstrap <<<

from datasets.eyecandies import EyeRGBDDataset
from torch.utils.data import DataLoader
import numpy as np
import random
from torchvision import transforms
from models.trd.encoder import ResNet50Encoder
from models.trd.decoder import ResNet50DualModalDecoder
from utils.losses import *
from eval.eval_utils import cal_anomaly_map
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score, average_precision_score
from eval.metrics_utils import calculate_au_pro


def setup_seed(seed):
    """固定随机种子保证可复现性（禁用 cudnn benchmark，启用 deterministic）。"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def f1_score_max(y_true, y_score):
    """在 PR 曲线上扫描所有阈值，返回能达到的最大 F1 分数。

    注意：``precision_recall_curve`` 未在文件顶部显式 import，依赖来自
    ``from utils.losses import *`` 间接暴露。如果切换环境报错需显式导入。
    """
    precs, recs, thrs = precision_recall_curve(y_true, y_score)

    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    f1s = f1s[:-1]  # 最后一个点对应无样本，丢弃
    return f1s.max()


def train_one_epoch(teacher_rgb, teacher_depth, student_rgb, student_depth, train_dataloader, optimizer_rgb, optimizer_depth, device, log, epoch):
    """双学生单 epoch 训练。

    每个 batch 依次：
        1. 前向两个冻结教师得到多尺度特征；
        2. student_rgb 用 (T_rgb, T_depth) 预测重建 RGB 教师特征；计算 4 项蒸馏损失；
        3. student_depth 用 (T_depth, T_rgb) 同理重建 Depth/Normal 教师特征；
        4. 两个优化器各自独立 step（彼此不共享参数）。
    """
    loss_rgb_list = []
    loss_depth_list = []
    for batch_idx, data in enumerate(train_dataloader, 0):
        rgb_image, depth_image, _, _, _ = data
        rgb_image = rgb_image.to(device)
        depth_image = depth_image.to(device)

        # 教师前向（冻结，不回传梯度）
        with torch.no_grad():
            output_Tr = teacher_rgb(rgb_image)
            output_Td = teacher_depth(depth_image)

            # 显式 detach 一份，避免在学生里被误用于梯度计算
            output_Tr_detach = [output_Tr[0].detach(), output_Tr[1].detach(), output_Tr[2].detach()]
            output_Td_detach = [output_Td[0].detach(), output_Td[1].detach(), output_Td[2].detach()]

        # ---- 学生 A：RGB 重建 ----
        # 输入：RGB 特征（主）+ Depth 特征（辅）
        # 输出：proj_d      - filter 投影
        #       proj_d_amply- amply 投影
        #       output_Sr   - 解码特征（融合后）
        #       output_Sr_am- 解码特征（未融合，纯解码版）
        proj_d, proj_d_amply, output_Sr, output_Sr_am = student_rgb(output_Tr_detach, output_Td_detach)
        # 四项蒸馏损失：解码结果、两路投影、纯解码版都对齐到 RGB 教师特征
        loss_rgb = loss_distil(output_Sr, output_Tr) + loss_distil(proj_d, output_Tr) + loss_distil(proj_d_amply, output_Tr) + loss_distil(output_Sr_am, output_Tr)
        loss_rgb_list.append(loss_rgb.item())
        optimizer_rgb.zero_grad()
        loss_rgb.backward()
        optimizer_rgb.step()

        # ---- 学生 B：Depth/Normal 重建（对称逻辑）----
        proj_r, proj_r_amply, output_Sd, output_Sd_am = student_depth(output_Td_detach, output_Tr_detach)
        loss_depth = loss_distil(output_Sd, output_Td) + loss_distil(proj_r, output_Td) + loss_distil(proj_r_amply, output_Td) + loss_distil(output_Sd_am, output_Td)
        loss_depth_list.append(loss_depth.item())
        optimizer_depth.zero_grad()
        loss_depth.backward()
        optimizer_depth.step()

    print('epoch %d, loss_rgb: %.10f, loss_depth: %.10f' % (epoch + 1, np.mean(loss_rgb_list), np.mean(loss_depth_list)))

    return


def train(device, classname, data_root, log, epochs, learning_rate, batch_size, img_size, ckp=None):
    """单个类别的端到端训练：构建数据 -> 模型 -> 训练 -> 验证归一化参数 -> 测试。"""
    if ckp is not None:
        ckp_path = ckp + classname + '.pth'
    else:
        ckp_path = None

    # ImageNet 标准均值/方差（预训练特征对应的归一化）
    train_mean = [0.485, 0.456, 0.406]
    train_std = [0.229, 0.224, 0.225]

    # RGB / 测试集使用相同的 transform；depth_transform 只做 Normalize（输入已是 Tensor）
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(train_mean, train_std)])
    depth_transform = transforms.Compose([
        transforms.Normalize(train_mean, train_std)])

    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(train_mean, train_std)])
    gt_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()])

    # Eyecandies 预处理目录要求（见 utils/preprocess_eyecandies.py）
    train_dir = data_root + classname + '/train'
    valid_dir = data_root + classname + '/validation'
    test_dir = data_root + classname + '/test'

    train_data = EyeRGBDDataset(data_dir=train_dir, transform=transform, depth_transform=depth_transform)
    valid_data = EyeRGBDDataset(data_dir=valid_dir, transform=transform, depth_transform=depth_transform)
    test_data = EyeRGBDDataset(data_dir=test_dir, transform=test_transform, depth_transform=depth_transform,
                                   test=True, gt_transform=gt_transform)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=16)
    valid_dataloader = DataLoader(valid_data, batch_size=1, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

    # ---- 构建两个教师（ImageNet 预训练，冻结）----
    teacher_rgb = ResNet50Encoder()
    teacher_rgb.to(device)
    teacher_rgb.eval()

    teacher_depth = ResNet50Encoder()
    teacher_depth.to(device)
    teacher_depth.eval()

    # ---- 构建两个学生（反向 ResNet + Projector + skip 融合）----
    student_depth = ResNet50DualModalDecoder(pretrained=False)
    student_depth.to(device)
    student_depth.train()

    student_rgb = ResNet50DualModalDecoder(pretrained=False)
    student_rgb.to(device)
    student_rgb.train()

    # 两个学生各自 Adam + 指数衰减 LR（常见的 RD 设置）
    optimizer_rgb = torch.optim.Adam(student_rgb.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizer_depth = torch.optim.Adam(student_depth.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    scheduler_rgb = torch.optim.lr_scheduler.ExponentialLR(optimizer_rgb, gamma=0.95)
    scheduler_depth = torch.optim.lr_scheduler.ExponentialLR(optimizer_depth, gamma=0.95)

    for epoch in range(epochs):
        student_rgb.train()
        student_depth.train()
        train_one_epoch(teacher_rgb, teacher_depth, student_rgb, student_depth, train_dataloader, optimizer_rgb, optimizer_depth, device, log, epoch)
        student_rgb.eval()
        student_depth.eval()

        scheduler_rgb.step()
        scheduler_depth.step()

    # 用验证集估计 RGB/Depth 两路异常图的均值/方差 —— 作为测试时的 z-score 参数
    params = valid(teacher_rgb, teacher_depth, student_rgb, student_depth, valid_dataloader, device)
    print(params)
    test(teacher_rgb, teacher_depth, student_rgb, student_depth, test_dataloader, device, log, epochs-1, classname, data_root, ckp_path, params)


def test(teacher_rgb, teacher_depth, student_rgb, student_depth, test_dataloader, device, log, epoch, classname, data_root, ckp_path, params=None):
    """测试循环 + 三种指标视角（融合 / 只看 RGB / 只看 Normal）的评估。"""
    # 如需保存 checkpoint
    if ckp_path is not None:
        torch.save({'student_rgb': student_rgb.state_dict(),
                    'student_normals': student_depth.state_dict()}, ckp_path)

    # 像素级 / 样本级 gt & 预测；后缀 _r / _d 分别是 RGB / Depth 单分支版本
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    pr_list_px_r = []
    pr_list_sp_r = []
    pr_list_px_d = []
    pr_list_sp_d = []

    # PRO 指标需要 2D 数组列表
    gts = []
    predictions = []
    predictions_r = []
    predictions_d = []

    with torch.no_grad():
        for i, data in enumerate(test_dataloader, 0):
            rgb_image, depth_image, gt, ad_label, ad_type = data
            rgb_img = rgb_image.to(device)
            depth_img = depth_image.to(device)

            # 双教师前向
            output_Trgb = teacher_rgb(rgb_img)
            output_Td = teacher_depth(depth_img)

            # 只取两个学生"融合后"的解码特征 output_Srgb / output_Sd（用于和教师对比）
            _, _, _, output_Srgb = student_rgb(output_Trgb, output_Td)
            _, _, _, output_Sd = student_depth(output_Td, output_Trgb)

            # 得到 RGB / Depth 分支的异常图（add 模式：累加三尺度余弦距离）
            anomaly_map_rgb, _ = cal_anomaly_map(output_Srgb, output_Trgb, amap_mode='add')
            anomaly_map_depth, _ = cal_anomaly_map(output_Sd, output_Td, amap_mode='add')

            # 用验证集统计的均值/方差做 z-score，让两路尺度可比后再相加
            if params is not None:
                anomaly_map_rgb = (anomaly_map_rgb - params[0]) / params[1]
                anomaly_map_depth = (anomaly_map_depth - params[2]) / params[3]

            anomaly_map = anomaly_map_rgb + anomaly_map_depth

            # 高斯滤波平滑（sigma=4 经验值），二值化 gt 以兼容 ToTensor 的线性插值
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
            pr_list_px.extend(anomaly_map.ravel())
            gt_list_sp.append(np.max(gt.cpu().numpy().astype(int)))      # 样本级 gt：图中有异常则 1
            pr_list_sp.append(np.max(anomaly_map))                       # 样本级分数：像素最大异常值
            gts.append(gt.squeeze().cpu().detach().numpy())
            predictions.append(anomaly_map)

            # 单分支结果同样记录（便于观察各模态贡献）
            anomaly_map_rgb = gaussian_filter(anomaly_map_rgb, sigma=4)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            pr_list_px_r.extend(anomaly_map_rgb.ravel())
            pr_list_sp_r.append(np.max(anomaly_map_rgb))
            predictions_r.append(anomaly_map_rgb)

            anomaly_map_depth = gaussian_filter(anomaly_map_depth, sigma=4)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            pr_list_px_d.extend(anomaly_map_depth.ravel())
            pr_list_sp_d.append(np.max(anomaly_map_depth))
            predictions_d.append(anomaly_map_depth)

        print('-----------------------testing %d epoch-----------------------' % (epoch + 1))

        # 分别打印三种视角的指标
        print('add:')
        test_metric(gt_list_px, pr_list_px, gt_list_sp, pr_list_sp, predictions, gts)

        print('rgb:')
        test_metric(gt_list_px, pr_list_px_r, gt_list_sp, pr_list_sp_r, predictions_r, gts)

        print('normal:')
        test_metric(gt_list_px, pr_list_px_d, gt_list_sp, pr_list_sp_d, predictions_d, gts)

    return


def test_metric(gt_list_px, pr_list_px, gt_list_sp, pr_list_sp, predictions, gts):
    """统一打印像素级/样本级 AUROC / AP / F1 + AU-PRO@{30%, 10%, 5%, 1%}。"""
    auroc_px = roc_auc_score(gt_list_px, pr_list_px)
    auroc_sp = roc_auc_score(gt_list_sp, pr_list_sp)

    ap_px = average_precision_score(gt_list_px, pr_list_px)
    ap_sp = average_precision_score(gt_list_sp, pr_list_sp)

    f1_px = f1_score_max(gt_list_px, pr_list_px)
    f1_sp = f1_score_max(gt_list_sp, pr_list_sp)

    # 默认 integration_limit=[0.3, 0.1, 0.05, 0.01]
    au_pros, _ = calculate_au_pro(gts, predictions)
    pro = au_pros[0]
    pro_10 = au_pros[1]
    pro_5 = au_pros[2]
    pro_1 = au_pros[3]

    print(" I-AUROC | P-AUROC | AUPRO@30% | AUPRO@10% | AUPRO@5% | AUPRO@1% |   I-AP   |   P-AP   |   I-F1   |   P-F1")
    print(
        f'  {auroc_sp:.3f}  |  {auroc_px:.3f}  |   {pro:.3f}   |   {pro_10:.3f}   |   {pro_5:.3f}  |   {pro_1:.3f}  |   {ap_sp:.3f}  |   {ap_px:.3f}  |   {f1_sp:.3f}  |   {f1_px:.3f}',
        end='\n')

    # 同时写入日志文件（若 log 有效）
    print(" I-AUROC | P-AUROC | AUPRO@30% | AUPRO@10% | AUPRO@5% | AUPRO@1% |   I-AP   |   P-AP   |   I-F1   |   P-F1", file=log)
    print(
        f'  {auroc_sp:.3f}  |  {auroc_px:.3f}  |   {pro:.3f}   |   {pro_10:.3f}   |   {pro_5:.3f}  |   {pro_1:.3f}  |   {ap_sp:.3f}  |   {ap_px:.3f}  |   {f1_sp:.3f}  |   {f1_px:.3f}',
        end='\n', file=log)


def valid(teacher_rgb, teacher_depth, student_rgb, student_depth, valid_dataloader, device):
    """在验证集上（全部为正常样本）统计 RGB / Depth 两路异常图的均值 + 方差。

    返回 [mean_r, std_r, mean_d, std_d]：用于 test 阶段做 z-score 归一化，
    使两路异常图量纲一致再相加融合。
    """
    a_rgb = []
    a_depth = []
    with torch.no_grad():
        for i, data in enumerate(valid_dataloader, 0):
            rgb_image, depth_image, _, _, _ = data
            rgb_img = rgb_image.to(device)
            depth_img = depth_image.to(device)

            with torch.no_grad():
                output_Trgb = teacher_rgb(rgb_img)
                output_Td = teacher_depth(depth_img)

                _, _, _, output_Srgb = student_rgb(output_Trgb, output_Td)
                _, _, _, output_Sd = student_depth(output_Td, output_Trgb)

                anomaly_map_rgb, _ = cal_anomaly_map(output_Srgb, output_Trgb, amap_mode='add')
                anomaly_map_depth, _ = cal_anomaly_map(output_Sd, output_Td, amap_mode='add')

                a_rgb.append(anomaly_map_rgb)
                a_depth.append(anomaly_map_depth)

    # 全图平均/标准差（所有像素一起统计）
    a_rgb_array = np.array(a_rgb)
    a_depth_array = np.array(a_depth)

    mean_r = np.mean(a_rgb_array)
    std_r = np.std(a_rgb_array)

    mean_d = np.mean(a_depth_array)
    std_d = np.std(a_depth_array)

    return [mean_r, std_r, mean_d, std_d]


if __name__ == "__main__":

    # Eyecandies 官方 10 个类别
    classnames = ['CandyCane', 'ChocolateCookie', 'ChocolatePraline', 'Confetto', 'GummyBear', 'HazelnutTruffle', 'LicoriceSandwich', 'Lollipop', 'Marshmallow', 'PeppermintCandy']

    # 训练超参（可按需修改）
    learning_rate = 0.005
    batch_size = 16
    img_size = 256
    data_root = './data/Eyecandies_preprocessed/'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log = None  # None 表示不写入日志文件；如需可改为 open('xxx.log', 'w')
    # ckp = './checkpoints/CRD_eye_rgb_normals_seed42/'   # 示例保存路径
    ckp = None

    # 逐类别训练 + 评估（每个类别重置种子，保证实验独立）
    for i in range(len(classnames)):
        setup_seed(42)
        classname = classnames[i]
        epochs_i = 100
        print('-----------------------training on ' + classname + '[%d / %d]-----------------------' % (
            i + 1, len(classnames)))
        train(device, classname, data_root, log, epochs_i, learning_rate, batch_size, img_size, ckp)
