import torch
# >>> path-bootstrap (auto-added on restructure) >>>
import os, sys
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)
# <<< path-bootstrap <<<

import os
from datasets.dataset_rgbn import MVTecADRGBNDataset
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
from sklearn.metrics import precision_recall_curve
from eval.metrics_utils import calculate_au_pro

# --------------------------------------------------------------------------------------
# 本脚本用途：
# 1) 在 MVTec3D 数据集上进行 RGB + Depth/Normal（此处简称 RGBN）的异常检测训练与评估。
# 2) 采用 Teacher-Student 蒸馏范式：Teacher 为冻结的特征提取器（ResNet50Encoder），
#    Student 为双模态解码/投影网络（ResNet50DualModalDecoder），学习拟合 Teacher 特征。
# 3) 测试阶段通过 student/teacher 特征差异生成 anomaly map，并计算 AUROC/AP/F1/AUPRO 等指标。
#
# 注意：本次仅“加注释”，不改动任何训练逻辑/超参/实现细节。
# --------------------------------------------------------------------------------------


def setup_seed(seed, deterministic=False):
    """设置随机种子，控制可复现性。

    Args:
        seed: 随机种子。
        deterministic: 是否强制使用确定性 cudnn 算法。
            - True：可复现性更强，但速度通常更慢。
            - False：启用 cudnn benchmark 自动选择最快卷积算法，训练更快但略影响完全复现。
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # deterministic=True 更可复现但更慢；训练提速建议 False
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def f1_score_max(y_true, y_score):
    """计算“最大 F1”（在不同阈值下的 F1 取最大值）。

    说明：
        precision_recall_curve 会返回一系列阈值 thrs 以及对应 precision/recall。
        这里遍历这些点计算 F1，并取最大值作为该分数序列的最佳 F1。

    Args:
        y_true: 0/1 标签（像素级或图像级）。
        y_score: 连续预测分数（异常分数）。

    Returns:
        最大 F1 值。
    """
    precs, recs, thrs = precision_recall_curve(y_true, y_score)

    # F1 = 2PR/(P+R)
    f1s = 2 * precs * recs / (precs + recs + 1e-7)

    # precision_recall_curve 的输出中，最后一个 precision/recall 点通常没有对应阈值
    f1s = f1s[:-1]
    return f1s.max()


def train_one_epoch(teacher_rgb, teacher_depth, student_rgb, student_depth, train_dataloader, optimizer_rgb, optimizer_depth, device, log, epoch, scaler=None):
    """单个 epoch 的训练。

    训练策略：
        - teacher_rgb/teacher_depth：冻结，仅前向提取多层特征。
        - student_rgb/student_depth：可训练，分别学习拟合对应 teacher 的特征。
        - 双优化器：RGB 分支与 Depth 分支分别反传与更新。
        - 支持 AMP：CUDA 下可用混合精度以提升速度/降低显存。

    Args:
        teacher_rgb: RGB Teacher（ResNet50Encoder）。
        teacher_depth: Depth/Normal Teacher（ResNet50Encoder）。
        student_rgb: RGB Student（ResNet50DualModalDecoder）。
        student_depth: Depth/Normal Student（ResNet50DualModalDecoder）。
        train_dataloader: 训练 DataLoader。
        optimizer_rgb/optimizer_depth: 两个分支各自的 optimizer。
        device: torch.device。
        log: 日志文件句柄。
        epoch: 当前 epoch（从 0 开始）。
        scaler: GradScaler（启用 AMP 时使用）。
    """
    loss_rgb_list = []
    loss_depth_list = []

    # 只有在 CUDA 且传入 scaler 时才启用 AMP
    use_amp = scaler is not None and device.type == "cuda"

    for batch_idx, data in enumerate(train_dataloader, 0):
        # Dataset 返回通常形如：(rgb, depth/normal, gt, label, type)
        rgb_image, depth_image, _, _, _ = data
        rgb_image = rgb_image.to(device, non_blocking=True)
        depth_image = depth_image.to(device, non_blocking=True)

        # Teacher 前向：不需要梯度
        with torch.no_grad():
            output_Tr = teacher_rgb(rgb_image)      # RGB 多层特征
            output_Td = teacher_depth(depth_image)  # Depth/Normal 多层特征

            # 取前 3 层特征并 detach（双保险，避免意外构图或引用梯度）
            output_Tr_detach = [output_Tr[0].detach(), output_Tr[1].detach(), output_Tr[2].detach()]
            output_Td_detach = [output_Td[0].detach(), output_Td[1].detach(), output_Td[2].detach()]

        # 清空梯度（两个分支分别更新）
        optimizer_rgb.zero_grad(set_to_none=True)
        optimizer_depth.zero_grad(set_to_none=True)

        if use_amp:
            # autocast：混合精度前向
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                # student_rgb 输入：RGB teacher 特征 + Depth teacher 特征
                # 返回：
                #   proj_d / proj_d_amply: 可能是来自 Depth 信息的投影/增强特征（具体取决于 decoder 实现）
                #   output_Sr / output_Sr_am: RGB 学生输出特征（及其增强/融合版本）
                proj_d, proj_d_amply, output_Sr, output_Sr_am = student_rgb(output_Tr_detach, output_Td_detach)

                # RGB 分支蒸馏损失：多路 student 输出都对齐到 RGB teacher 的特征
                loss_rgb = (
                    loss_distil(output_Sr, output_Tr)
                    + loss_distil(proj_d, output_Tr)
                    + loss_distil(proj_d_amply, output_Tr)
                    + loss_distil(output_Sr_am, output_Tr)
                )
    
                # student_depth 输入：Depth teacher 特征 + RGB teacher 特征
                proj_r, proj_r_amply, output_Sd, output_Sd_am = student_depth(output_Td_detach, output_Tr_detach)

                # Depth 分支蒸馏损失：多路 student 输出都对齐到 Depth teacher 的特征
                loss_depth = (
                    loss_distil(output_Sd, output_Td)
                    + loss_distil(proj_r, output_Td)
                    + loss_distil(proj_r_amply, output_Td)
                    + loss_distil(output_Sd_am, output_Td)
                )

            # AMP 反传与更新（两个 optimizer 分别 step）
            scaler.scale(loss_rgb).backward()
            scaler.step(optimizer_rgb)

            scaler.scale(loss_depth).backward()
            scaler.step(optimizer_depth)

            scaler.update()
        else:
            # 非 AMP：常规 float32 训练
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

        # 记录 loss，用于 epoch 平均
        loss_rgb_list.append(loss_rgb.item())
        loss_depth_list.append(loss_depth.item())

    # 输出 epoch 平均 loss
    print('epoch %d, loss_rgb: %.10f, loss_depth: %.10f' % (epoch + 1, np.mean(loss_rgb_list), np.mean(loss_depth_list)))
    print(
        'epoch %d, loss_rgb: %.10f, loss_depth: %.10f' % (epoch + 1, np.mean(loss_rgb_list), np.mean(loss_depth_list)), file=log)

    return


def train(device, classname, data_root, log, epochs, learning_rate, batch_size, img_size, ckp):
    """针对单个类别进行训练，并在最后进行 validation 统计与 test 评估。

    Args:
        device: torch.device。
        classname: 类别名（MVTec3D 子类）。
        data_root: 数据集根目录（末尾带 os.sep）。
        log: 日志文件句柄。
        epochs: 训练轮数。
        learning_rate: 学习率。
        batch_size: 批大小。
        img_size: 输入分辨率（正方形）。
        ckp: checkpoint 保存目录前缀（None 表示不保存）。
    """
    if ckp is not None:
        ckp_path = ckp + classname + '.pth'
    else:
        ckp_path = None

    # 归一化参数（ImageNet 均值方差）
    train_mean = [0.485, 0.456, 0.406]
    train_std = [0.229, 0.224, 0.225]

    # RGB 预处理：Resize -> ToTensor -> Normalize
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(train_mean, train_std)])

    # Depth/Normal 预处理：Resize -> Normalize
    # 注意：此处没有 ToTensor，意味着 dataset 内部很可能已经输出 Tensor（否则会类型不匹配）
    depth_transform = transforms.Compose([
        transforms.Resize((img_size, img_size), antialias=True),
        transforms.Normalize(train_mean, train_std)])

    # test 的 RGB 预处理
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(train_mean, train_std)])

    # GT mask 预处理：Resize -> ToTensor（不 normalize）
    gt_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()])

    # 数据路径（train/validation/test）
    train_dir = data_root + classname + '/train'
    valid_dir = data_root + classname + '/validation'
    test_dir = data_root + classname + '/test'

    # 某些类使用不同的 k（可能与几何/邻域处理相关，具体见 dataset 实现）
    k_dict = {'cookie': 3, 'dowel' : 7, 'foam' : 7, 'tire' : 7}
    if classname in k_dict:
        k = k_dict[classname]
    else:
        k = 5
    print(k)
    print(k, file=log)

    # 构建数据集
    train_data = MVTecADRGBNDataset(data_dir=train_dir, transform=transform, depth_transform=depth_transform, k=k)
    valid_data = MVTecADRGBNDataset(data_dir=valid_dir, transform=transform, depth_transform=depth_transform, k=k)
    test_data = MVTecADRGBNDataset(data_dir=test_dir, transform=test_transform, depth_transform=depth_transform,
                                   test=True, gt_transform=gt_transform, k=k)

    # DataLoader：通过多进程/预取提升吞吐
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
        num_workers=8,
        persistent_workers=True,
        prefetch_factor=2,
    )
    valid_dataloader = DataLoader(valid_data, batch_size=1, shuffle=False, pin_memory=(device.type == "cuda"))
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, pin_memory=(device.type == "cuda"))

    # ------------------------------ Teacher（冻结）------------------------------
    teacher_rgb = ResNet50Encoder()
    teacher_rgb.to(device)
    teacher_rgb.eval()
    for p in teacher_rgb.parameters():
        p.requires_grad_(False)

    teacher_depth = ResNet50Encoder()
    teacher_depth.to(device)
    teacher_depth.eval()
    for p in teacher_depth.parameters():
        p.requires_grad_(False)

    # ------------------------------ Student（可训练）------------------------------
    student_depth = ResNet50DualModalDecoder(pretrained=False)
    student_depth.to(device)
    student_depth.train()

    student_rgb = ResNet50DualModalDecoder(pretrained=False)
    student_rgb.to(device)
    student_rgb.train()

    # 两个分支各自使用 Adam
    optimizer_rgb = torch.optim.Adam(student_rgb.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizer_depth = torch.optim.Adam(student_depth.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    # 指数衰减 LR
    scheduler_rgb = torch.optim.lr_scheduler.ExponentialLR(optimizer_rgb, gamma=0.95)
    scheduler_depth = torch.optim.lr_scheduler.ExponentialLR(optimizer_depth, gamma=0.95)

    # AMP scaler（仅 CUDA 有效）
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # ------------------------------ 训练循环 ------------------------------
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

    # 在 validation 集上统计 anomaly_map 的均值与方差（用于 test 阶段标准化）
    params = valid(teacher_rgb, teacher_depth, student_rgb, student_depth, valid_dataloader, device)
    print(params)
    print(params,file=log)

    # 测试评估
    test(teacher_rgb, teacher_depth, student_rgb, student_depth, test_dataloader, device, log, epochs - 1, classname,
         data_root, ckp_path, params)


def test(teacher_rgb, teacher_depth, student_rgb, student_depth, test_dataloader, device, log, epoch, classname, data_root, ckp_path, params=None):
    """在 test 集上评估并打印指标。

    核心步骤：
        1) teacher/student 前向得到多层特征
        2) 由 cal_anomaly_map 生成 RGB/Depth 的像素级异常图
        3) 可选地用 validation 的均值/方差进行标准化（Z-score）
        4) RGB 与 Depth 异常图相加融合，并做高斯平滑
        5) 计算像素级/图像级 AUROC, AP, F1-max 以及 AUPRO

    Args:
        params: valid() 返回的 [mean_r, std_r, mean_d, std_d]；若为 None 则不标准化。
    """
    # 保存 checkpoint（仅保存 student 参数）
    if ckp_path is not None:
        torch.save({'student_rgb': student_rgb.state_dict(),
                'student_depth': student_depth.state_dict()}, ckp_path)

    # 像素级指标（flatten 后的全量像素标签/分数）
    gt_list_px = []
    pr_list_px = []

    # 图像级指标（每张图取 max 像素作为图像分数）
    gt_list_sp = []
    pr_list_sp = []

    # 分别统计 RGB-only 与 Depth-only
    pr_list_px_r = []
    pr_list_sp_r = []
    pr_list_px_d = []
    pr_list_sp_d = []

    # AUPRO 需要整张二维 mask 与 anomaly map
    gts = []
    predictions = []
    predictions_r = []
    predictions_d = []

    with torch.no_grad():
        for i, data in enumerate(test_dataloader, 0):
            # ad_label/ad_type 在本函数中未参与计算，仅用于上层可能记录
            rgb_image, depth_image, gt, ad_label, ad_type = data
            rgb_img = rgb_image.to(device)
            depth_img = depth_image.to(device)

            # teacher 特征
            output_Trgb = teacher_rgb(rgb_img)
            output_Td = teacher_depth(depth_img)

            # student 输出：此处只使用返回的第 4 个输出（output_S*）用于 anomaly map
            _, _, _, output_Srgb = student_rgb(output_Trgb, output_Td)
            _, _, _, output_Sd = student_depth(output_Td, output_Trgb)

            # 根据 student 与 teacher 的特征差异生成像素级异常图
            anomaly_map_rgb, _ = cal_anomaly_map(output_Srgb, output_Trgb, out_size=img_size, amap_mode='add')
            anomaly_map_depth, _ = cal_anomaly_map(output_Sd, output_Td, out_size=img_size, amap_mode='add')

            # 可选标准化：减少不同模态/类别之间分数尺度差异
            if params is not None:
                anomaly_map_rgb = (anomaly_map_rgb - params[0]) / params[1]
                anomaly_map_depth = (anomaly_map_depth - params[2]) / params[3]

            # 融合：RGB + Depth
            anomaly_map = anomaly_map_rgb + anomaly_map_depth

            # 平滑：抑制噪点，使异常区域更连贯
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)

            # GT 二值化（确保为 0/1）
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0

            # 像素级：展开为一维列表
            gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
            pr_list_px.extend(anomaly_map.ravel())

            # 图像级：每张图取最大像素
            gt_list_sp.append(np.max(gt.cpu().numpy().astype(int)))
            pr_list_sp.append(np.max(anomaly_map))

            # AUPRO 需要二维
            gts.append(gt.squeeze().cpu().detach().numpy())  # (H,W)
            predictions.append(anomaly_map)  # (H,W)

            # ----------------- RGB-only -----------------
            anomaly_map_rgb = gaussian_filter(anomaly_map_rgb, sigma=4)
            pr_list_px_r.extend(anomaly_map_rgb.ravel())
            pr_list_sp_r.append(np.max(anomaly_map_rgb))
            predictions_r.append(anomaly_map_rgb)  # (H,W)

            # ----------------- Depth-only -----------------
            anomaly_map_depth = gaussian_filter(anomaly_map_depth, sigma=4)
            pr_list_px_d.extend(anomaly_map_depth.ravel())
            pr_list_sp_d.append(np.max(anomaly_map_depth))
            predictions_d.append(anomaly_map_depth)  # (H,W)

        print('-----------------------testing %d epoch-----------------------' % (epoch + 1))
        print('-----------------------testing %d epoch-----------------------' % (epoch + 1), file=log)

        # 分别输出融合、RGB-only、Depth-only 的指标
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
    """计算并打印评估指标。

    指标说明：
        - P-AUROC: Pixel-level AUROC（像素级）
        - I-AUROC: Image-level AUROC（图像级：每张图取 max 像素作为图像分数）
        - P-AP/I-AP: Average Precision
        - P-F1/I-F1: 最大 F1（对所有阈值取最大）
        - AUPRO@x%: PRO 曲线在指定 FPR 上限下的面积（实现见 calculate_au_pro）
    """
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
    """在 validation 集上统计 anomaly map 的分布（均值/标准差）。

    目的：
        test 阶段可用该均值/标准差对 anomaly_map 做标准化：
            (map - mean) / std
        以减少不同模态输出尺度差异。

    Returns:
        [mean_r, std_r, mean_d, std_d]
    """
    a_rgb = []
    a_depth = []
    with torch.no_grad():
        for i, data in enumerate(valid_dataloader, 0):
            rgb_image, depth_image, _, _, _ = data
            rgb_img = rgb_image.to(device)
            depth_img = depth_image.to(device)

            # no_grad 已经包裹，这里再嵌套一层不影响正确性
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

    # 训练类别列表（MVTec3D 官方 10 类）
    # 4060 Laptop 8G 推荐：RGBN batch=8，workers=8（Windows 下 16 往往更慢）
    classnames = ['bagel', 'cable_gland', 'carrot', 'cookie', 'dowel', 'foam', 'peach', 'potato', 'rope', 'tire']

    # 超参数
    learning_rate = 0.005
    batch_size = 8
    num_workers = 8  # 注意：当前 DataLoader 内写死 num_workers=8，这里仅作配置占位
    img_size = 256

    # 数据根目录（确保末尾有分隔符）
    data_root = r'F:\TRD\data\mvtec_3d_anomaly_detection' + os.sep

    # 设备选择
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 日志文件
    os.makedirs("./logs", exist_ok=True)
    log = open("./logs/CRD_mvtec3d_rgb_normal_seed42_200e.txt", 'a')

    # checkpoint 目录前缀；None 表示不保存
    ckp = None

    # 逐类别训练与评估
    for i in range(len(classnames)):
        setup_seed(42, deterministic=False)
        classname = classnames[i]
        epochs_i = 200

        print('-----------------------training on ' + classname + '[%d / %d]-----------------------' % (
            i + 1, len(classnames)))
        print('-----------------------training on ' + classname + '[%d / %d]-----------------------' % (
            i + 1, len(classnames)), file=log)

        train(device, classname, data_root, log, epochs_i, learning_rate, batch_size, img_size, ckp)
