# -*- coding: utf-8 -*-
"""
MVTec 3D-AD 数据集上的 TRD 双模态训练脚本（主模态 RGB + 辅助模态深度图）。

核心差异对比 ``train_Eyecandies.py``：
    - 数据集：``MVTecADRGBDDataset`` 会在 __getitem__ 里做 tiff 读取、平面分割、
      深度归一化、缺失填充（见 dataset/dataset_rgbd.py）。
    - 训练稳定性/速度增强：
        * 支持 ``torch.cuda.amp`` 混合精度训练（scaler）
        * DataLoader 走 ``persistent_workers`` + ``prefetch_factor``，减少 Windows 下的启动开销
        * 教师参数显式 ``requires_grad_(False)`` 以节省显存
    - 日志：print 同时写入 log 文件（CRD_mvtec3d_rgb_depth_seed42.txt）。

整体训练流程和双学生互蒸馏思路同 ``train_Eyecandies.py``，此处不再重复。
"""

import torch
# >>> path-bootstrap (auto-added on restructure) >>>
import os, sys
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)
# <<< path-bootstrap <<<

import os
from datasets.dataset_rgbd import MVTecADRGBDDataset                 # 提供 RGBD 数据
from torch.utils.data import DataLoader
import numpy as np
import random
from torchvision import transforms
from models.trd.encoder import ResNet50Encoder                         # 教师（编码器）
from models.trd.decoder import ResNet50DualModalDecoder                # 学生（双模态解码器）
from utils.losses import *                                          # 引入多种蒸馏损失
from eval.eval_utils import cal_anomaly_map                   # 特征差 -> 异常热图
from scipy.ndimage import gaussian_filter                           # 热图高斯平滑
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import precision_recall_curve
from eval.metrics_utils import calculate_au_pro


def setup_seed(seed, deterministic=False):
    """统一设置各框架随机种子。

    deterministic=True：开启确定性模式（cudnn benchmark 关），可复现但训练变慢；
    deterministic=False：开启 benchmark 自动选最快卷积算法，速度优先。
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # deterministic=True 更可复现但更慢；训练提速建议 False
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def f1_score_max(y_true, y_score):
    """扫描 PR 曲线所有阈值，返回最大 F1。用来评估样本级/像素级分类最佳性能。"""
    precs, recs, thrs = precision_recall_curve(y_true, y_score)

    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    f1s = f1s[:-1]  # 丢最后一个空点
    return f1s.max()


def train_one_epoch(teacher_rgb, teacher_depth, student_rgb, student_depth, train_dataloader, optimizer_rgb, optimizer_depth, device, log, epoch, scaler=None):
    """双学生一个 epoch 的训练，支持 AMP 混合精度。

    scaler 非 None 且 device 为 CUDA 时走 AMP 分支（autocast + GradScaler）；
    否则退化为常规 FP32 训练。两个学生顺序各自计算 loss 并独立反向。
    """
    loss_rgb_list = []
    loss_depth_list = []

    use_amp = scaler is not None and device.type == "cuda"

    for batch_idx, data in enumerate(train_dataloader, 0):
        rgb_image, depth_image, _, _, _ = data
        rgb_image = rgb_image.to(device, non_blocking=True)  # pin_memory + non_blocking 加速 H2D
        depth_image = depth_image.to(device, non_blocking=True)

        # 教师冻结前向
        with torch.no_grad():
            output_Tr = teacher_rgb(rgb_image)
            output_Td = teacher_depth(depth_image)

            output_Tr_detach = [output_Tr[0].detach(), output_Tr[1].detach(), output_Tr[2].detach()]
            output_Td_detach = [output_Td[0].detach(), output_Td[1].detach(), output_Td[2].detach()]

        # set_to_none=True 比 zero_() 更省一次内存写入
        optimizer_rgb.zero_grad(set_to_none=True)
        optimizer_depth.zero_grad(set_to_none=True)

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                proj_d, proj_d_amply, output_Sr, output_Sr_am = student_rgb(output_Tr_detach, output_Td_detach)
                loss_rgb = (
                    loss_distil(output_Sr, output_Tr)
                    + loss_distil(proj_d, output_Tr)
                    + loss_distil(proj_d_amply, output_Tr)
                    + loss_distil(output_Sr_am, output_Tr)
                )

                proj_r, proj_r_amply, output_Sd, output_Sd_am = student_depth(output_Td_detach, output_Tr_detach)
                loss_depth = (
                    loss_distil(output_Sd, output_Td)
                    + loss_distil(proj_r, output_Td)
                    + loss_distil(proj_r_amply, output_Td)
                    + loss_distil(output_Sd_am, output_Td)
                )

            scaler.scale(loss_rgb).backward()
            scaler.step(optimizer_rgb)

            scaler.scale(loss_depth).backward()
            scaler.step(optimizer_depth)

            scaler.update()
        else:
            proj_d, proj_d_amply, output_Sr, output_Sr_am = student_rgb(output_Tr_detach, output_Td_detach)
            loss_rgb = (
                loss_distil(output_Sr, output_Tr)
                + loss_distil(proj_d, output_Tr)
                + loss_distil(proj_d_amply, output_Tr)
                + loss_distil(output_Sr_am, output_Tr)
            )
            loss_rgb.backward()
            optimizer_rgb.step()

            proj_r, proj_r_amply, output_Sd, output_Sd_am = student_depth(output_Td_detach, output_Tr_detach)
            loss_depth = (
                loss_distil(output_Sd, output_Td)
                + loss_distil(proj_r, output_Td)
                + loss_distil(proj_r_amply, output_Td)
                + loss_distil(output_Sd_am, output_Td)
            )
            loss_depth.backward()
            optimizer_depth.step()

        loss_rgb_list.append(loss_rgb.item())
        loss_depth_list.append(loss_depth.item())

    print('epoch %d, loss_rgb: %.10f, loss_depth: %.10f' % (epoch + 1, np.mean(loss_rgb_list), np.mean(loss_depth_list)))
    print(
        'epoch %d, loss_rgb: %.10f, loss_depth: %.10f' % (epoch + 1, np.mean(loss_rgb_list), np.mean(loss_depth_list)), file=log)

    return


def train(device, classname, data_root, log, epochs, learning_rate, batch_size, img_size, ckp):
    if ckp is not None:
        ckp_path = ckp + classname + '.pth'
    else:
        ckp_path = None

    train_mean = [0.485, 0.456, 0.406]
    train_std = [0.229, 0.224, 0.225]

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

    train_dir = data_root + classname + '/train'
    valid_dir = data_root + classname + '/validation'
    test_dir = data_root + classname + '/test'

    train_data = MVTecADRGBDDataset(data_dir=train_dir, transform=transform, depth_transform=depth_transform)
    valid_data = MVTecADRGBDDataset(data_dir=valid_dir, transform=transform, depth_transform=depth_transform)
    test_data = MVTecADRGBDDataset(data_dir=test_dir, transform=test_transform, depth_transform=depth_transform,
                                   test=True, gt_transform=gt_transform)
    # DataLoader: 提升吞吐（Windows 建议 4~8；过大反而慢）
    # 通过闭包读取 __main__ 中定义的 num_workers；若未定义则回退 8
    try:
        nw = num_workers  # 来自 __main__ 配置
    except NameError:
        nw = 8

    # persistent_workers=True 可以在 epoch 间复用 worker 进程，省去重启开销
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
        num_workers=nw,
        persistent_workers=(nw > 0),
        prefetch_factor=2,
    )
    # 验证/测试 batch_size=1，避免最后一个 batch 大小不一致；num_workers=0 足够
    valid_dataloader = DataLoader(valid_data, batch_size=1, shuffle=False, pin_memory=(device.type == "cuda"), num_workers=0)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, pin_memory=(device.type == "cuda"), num_workers=0)

    # ---- 教师（冻结参数 + eval 模式）----
    teacher_rgb = ResNet50Encoder()
    teacher_rgb.to(device)
    teacher_rgb.eval()
    for p in teacher_rgb.parameters():
        p.requires_grad_(False)  # 省显存：不保留 autograd 图

    teacher_depth = ResNet50Encoder()
    teacher_depth.to(device)
    teacher_depth.eval()
    for p in teacher_depth.parameters():
        p.requires_grad_(False)

    # ---- 双学生解码器 ----
    student_depth = ResNet50DualModalDecoder(pretrained=False)
    student_depth.to(device)
    student_depth.train()

    student_rgb = ResNet50DualModalDecoder(pretrained=False)
    student_rgb.to(device)
    student_rgb.train()

    optimizer_rgb = torch.optim.Adam(student_rgb.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizer_depth = torch.optim.Adam(student_depth.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    scheduler_rgb = torch.optim.lr_scheduler.ExponentialLR(optimizer_rgb, gamma=0.95)
    scheduler_depth = torch.optim.lr_scheduler.ExponentialLR(optimizer_depth, gamma=0.95)

    # 混合精度缩放器（CPU 下 enabled=False 会变成 no-op）
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    for epoch in range(epochs):
        student_rgb.train()
        student_depth.train()
        train_one_epoch(
            teacher_rgb,
            teacher_depth,
            student_rgb,
            student_depth,
            train_dataloader,
            optimizer_rgb,
            optimizer_depth,
            device,
            log,
            epoch,
            scaler=scaler,
        )
        student_rgb.eval()
        student_depth.eval()

        scheduler_rgb.step()
        scheduler_depth.step()

    params = valid(teacher_rgb, teacher_depth, student_rgb, student_depth, valid_dataloader, device)
    print(params)
    print(params,file=log)
    test(teacher_rgb, teacher_depth, student_rgb, student_depth, test_dataloader, device, log, epochs - 1, classname,
         data_root, ckp_path, params)


def test(teacher_rgb, teacher_depth, student_rgb, student_depth, test_dataloader, device, log, epoch, classname, data_root, ckp_path, params=None):
    if ckp_path is not None:
        torch.save({'student_rgb': student_rgb.state_dict(),
                'student_depth': student_depth.state_dict()}, ckp_path)

    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    pr_list_px_r = []
    pr_list_sp_r = []
    pr_list_px_d = []
    pr_list_sp_d = []
    gts = []
    predictions = []
    predictions_r = []
    predictions_d = []

    with torch.no_grad():
        for i, data in enumerate(test_dataloader, 0):
            rgb_image, depth_image, gt, ad_label, ad_type = data
            rgb_img = rgb_image.to(device)
            depth_img = depth_image.to(device)

            output_Trgb = teacher_rgb(rgb_img)
            output_Td = teacher_depth(depth_img)

            _, _, _, output_Srgb = student_rgb(output_Trgb, output_Td)
            _, _, _, output_Sd = student_depth(output_Td, output_Trgb)

            anomaly_map_rgb, _ = cal_anomaly_map(output_Srgb, output_Trgb, out_size=img_size, amap_mode='add')
            anomaly_map_depth, _ = cal_anomaly_map(output_Sd, output_Td, out_size=img_size, amap_mode='add')

            if params is not None:
                anomaly_map_rgb = (anomaly_map_rgb - params[0]) / params[1]
                anomaly_map_depth = (anomaly_map_depth - params[2]) / params[3]

            anomaly_map = anomaly_map_rgb + anomaly_map_depth

            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
            pr_list_px.extend(anomaly_map.ravel())
            gt_list_sp.append(np.max(gt.cpu().numpy().astype(int)))
            pr_list_sp.append(np.max(anomaly_map))
            gts.append(gt.squeeze().cpu().detach().numpy())  # * (256,256)
            predictions.append(anomaly_map)  # * (256,256)

            anomaly_map_rgb = gaussian_filter(anomaly_map_rgb, sigma=4)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            pr_list_px_r.extend(anomaly_map_rgb.ravel())
            pr_list_sp_r.append(np.max(anomaly_map_rgb))
            predictions_r.append(anomaly_map_rgb)  # * (256,256)

            anomaly_map_depth = gaussian_filter(anomaly_map_depth, sigma=4)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            pr_list_px_d.extend(anomaly_map_depth.ravel())
            pr_list_sp_d.append(np.max(anomaly_map_depth))
            predictions_d.append(anomaly_map_depth)  # * (256,256)

        print('-----------------------testing %d epoch-----------------------' % (epoch + 1))
        print('-----------------------testing %d epoch-----------------------' % (epoch + 1), file=log)


        print('add:')
        print('add:', file=log)
        test_metric(gt_list_px, pr_list_px, gt_list_sp, pr_list_sp, predictions, gts)

        print('rgb:')
        print('rgb:', file=log)
        test_metric(gt_list_px, pr_list_px_r, gt_list_sp, pr_list_sp_r, predictions_r, gts)

        print('depth:')
        print('depth:', file=log)
        test_metric(gt_list_px, pr_list_px_d, gt_list_sp, pr_list_sp_d, predictions_d, gts)


    return


def test_metric(gt_list_px, pr_list_px, gt_list_sp, pr_list_sp, predictions, gts):
    auroc_px = roc_auc_score(gt_list_px, pr_list_px)
    auroc_sp = roc_auc_score(gt_list_sp, pr_list_sp)

    ap_px = average_precision_score(gt_list_px, pr_list_px)
    ap_sp = average_precision_score(gt_list_sp, pr_list_sp)

    f1_px = f1_score_max(gt_list_px, pr_list_px)
    f1_sp = f1_score_max(gt_list_sp, pr_list_sp)

    au_pros, _ = calculate_au_pro(gts, predictions)
    pro = au_pros[0]
    pro_10 = au_pros[1]
    pro_5 = au_pros[2]
    pro_1 = au_pros[3]

    print(" I-AUROC | P-AUROC | AUPRO@30% | AUPRO@10% | AUPRO@5% | AUPRO@1% |   I-AP   |   P-AP   |   I-F1   |   P-F1")
    print(
        f'  {auroc_sp:.3f}  |  {auroc_px:.3f}  |   {pro:.3f}   |   {pro_10:.3f}   |   {pro_5:.3f}  |   {pro_1:.3f}  |   {ap_sp:.3f}  |   {ap_px:.3f}  |   {f1_sp:.3f}  |   {f1_px:.3f}',
        end='\n')

    print(" I-AUROC | P-AUROC | AUPRO@30% | AUPRO@10% | AUPRO@5% | AUPRO@1% |   I-AP   |   P-AP   |   I-F1   |   P-F1", file=log)
    print(
        f'  {auroc_sp:.3f}  |  {auroc_px:.3f}  |   {pro:.3f}   |   {pro_10:.3f}   |   {pro_5:.3f}  |   {pro_1:.3f}  |   {ap_sp:.3f}  |   {ap_px:.3f}  |   {f1_sp:.3f}  |   {f1_px:.3f}',
        end='\n', file=log)


def valid(teacher_rgb, teacher_depth, student_rgb, student_depth, valid_dataloader, device):
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

                anomaly_map_rgb, _ = cal_anomaly_map(output_Srgb, output_Trgb, out_size=img_size, amap_mode='add')
                anomaly_map_depth, _ = cal_anomaly_map(output_Sd, output_Td, out_size=img_size, amap_mode='add')

                a_rgb.append(anomaly_map_rgb)
                a_depth.append(anomaly_map_depth)

    a_rgb_array = np.array(a_rgb)
    a_depth_array = np.array(a_depth)

    mean_r = np.mean(a_rgb_array)
    std_r = np.std(a_rgb_array)

    mean_d = np.mean(a_depth_array)
    std_d = np.std(a_depth_array)

    return [mean_r, std_r, mean_d, std_d]


if __name__ == "__main__":

    setup_seed(111, deterministic=False)  # 全局种子；进入每个类别再重置一次

    # MVTec 3D-AD 的 10 个类别
    classnames = ['bagel', 'cable_gland', 'carrot', 'cookie', 'dowel', 'foam', 'peach', 'potato', 'rope', 'tire']

    # 训练超参配置
    learning_rate = 0.005
    batch_size = 12                 # 12 较为保守，根据显存可调大
    num_workers = 8                 # DataLoader 进程数（Windows 下不宜过大）
    img_size = 256
    data_root = '../data/mvtec_3d_anomaly_detection/'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 日志文件（append 模式，便于多次运行累积）
    os.makedirs("./logs", exist_ok=True)
    log = open("./logs/CRD_mvtec3d_rgb_depth_seed42.txt",'a')
    ckp = None  # 设为目录可保存 checkpoint

    for i in range(len(classnames)):
        # 每个类别重置同一种子保证可比较
        setup_seed(42, deterministic=False)
        classname = classnames[i]
        epochs_i = 200
        print('-----------------------training on ' + classname + '[%d / %d]-----------------------' % (
            i + 1, len(classnames)))
        print('-----------------------training on ' + classname + '[%d / %d]-----------------------' % (
            i + 1, len(classnames)), file=log)
        train(device, classname, data_root, log, epochs_i, learning_rate, batch_size, img_size, ckp)
