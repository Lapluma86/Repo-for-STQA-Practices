# -*- coding: utf-8 -*-
"""
钢轨数据集上的 TRD 双模态训练脚本（RGB + Depth）。

基于 MVTec 3D-AD 训练脚本改编，针对钢轨数据的特点：
    - 超长条形图像（6000×900）→ Patch 滑窗切分（7个 900×900 patch）
    - RGB 是 3 通道彩色图（不是灰度）
    - Depth 是 uint16 的 tiff 格式
    - 训练集：所有非 manifest 样本（假设为正常）
    - 测试集：manifest 中的标注样本（异常）
"""

# >>> path-bootstrap >>>
import os, sys
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)
# <<< path-bootstrap <<<

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import random
from datetime import datetime
from typing import Dict
from contextlib import nullcontext
from tqdm import tqdm

from datasets.rail_dataset import RailDualModalDataset
from models.trd.encoder import ResNet50Encoder
from models.trd.decoder import ResNet50DualModalDecoder
from utils.losses import loss_distil
from eval.eval_utils import cal_anomaly_map
from sklearn.metrics import roc_auc_score

# AMP（混合精度训练）
# 优先使用新的 torch.amp API（PyTorch >= 2.0），失败则回退到旧 API
try:
    from torch.amp import autocast as _autocast_new
    from torch.amp import GradScaler as _GradScalerNew

    def autocast(enabled=True, dtype=torch.float16):
        return _autocast_new("cuda", enabled=enabled, dtype=dtype)

    def make_scaler(enabled=True):
        # bf16 不需要 GradScaler；fp16 才需要
        return _GradScalerNew("cuda", enabled=enabled)

    AMP_AVAILABLE = True
except ImportError:
    try:
        from torch.cuda.amp import autocast as _autocast_old
        from torch.cuda.amp import GradScaler as _GradScalerOld

        def autocast(enabled=True, dtype=torch.float16):
            return _autocast_old(enabled=enabled, dtype=dtype)

        def make_scaler(enabled=True):
            return _GradScalerOld(enabled=enabled)

        AMP_AVAILABLE = True
    except ImportError:
        AMP_AVAILABLE = False


def setup_seed(seed, deterministic=False):
    """统一设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    # TF32 加速（Ampere+ GPU），矩阵乘法快 2-3×，几乎无损精度
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def train_one_epoch(teacher_rgb, teacher_depth, student_rgb, student_depth,
                    train_loader, optimizer_rgb, optimizer_depth,
                    device, epoch, log_file, scaler=None, amp_dtype=None,
                    channels_last=False):
    """训练一个 epoch（AMP + 双学生并行前向）

    amp_dtype: None=fp32, torch.float16=AMP fp16(需 scaler), torch.bfloat16=AMP bf16(不需 scaler)
    """
    student_rgb.train()
    student_depth.train()

    use_amp = amp_dtype is not None
    use_scaler = use_amp and amp_dtype == torch.float16

    loss_rgb_list = []
    loss_depth_list = []

    pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                desc=f'Epoch {epoch}', leave=False)

    for batch_idx, data in pbar:
        rgb = data["intensity"].to(device, non_blocking=True)
        depth = data["depth"].to(device, non_blocking=True)
        if channels_last:
            rgb = rgb.contiguous(memory_format=torch.channels_last)
            depth = depth.contiguous(memory_format=torch.channels_last)

        amp_ctx = autocast(enabled=use_amp, dtype=amp_dtype) if use_amp else nullcontext()

        with torch.no_grad():
            with amp_ctx:
                feat_t_rgb = teacher_rgb(rgb)
                feat_t_depth = teacher_depth(depth)
            feat_t_rgb = [f.detach() for f in feat_t_rgb]
            feat_t_depth = [f.detach() for f in feat_t_depth]

        optimizer_rgb.zero_grad(set_to_none=True)
        optimizer_depth.zero_grad(set_to_none=True)

        # 两个学生在同一 autocast 上下文中前向
        with (autocast(enabled=use_amp, dtype=amp_dtype) if use_amp else nullcontext()):
            proj_d, proj_d_amply, feat_s_rgb, feat_s_rgb_am = student_rgb(feat_t_rgb, feat_t_depth)
            loss_rgb = (
                loss_distil(feat_s_rgb, feat_t_rgb) +
                loss_distil(proj_d, feat_t_rgb) +
                loss_distil(proj_d_amply, feat_t_rgb) +
                loss_distil(feat_s_rgb_am, feat_t_rgb)
            )

            proj_r, proj_r_amply, feat_s_depth, feat_s_depth_am = student_depth(feat_t_depth, feat_t_rgb)
            loss_depth = (
                loss_distil(feat_s_depth, feat_t_depth) +
                loss_distil(proj_r, feat_t_depth) +
                loss_distil(proj_r_amply, feat_t_depth) +
                loss_distil(feat_s_depth_am, feat_t_depth)
            )

        # 总 loss 一次 backward（减少一次 Python→CUDA 调度）
        total_loss = loss_rgb + loss_depth
        if use_scaler:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer_rgb)
            scaler.step(optimizer_depth)
            scaler.update()
        else:
            # bf16 或 fp32：直接 step（bf16 范围与 fp32 相同，不会溢出）
            total_loss.backward()
            optimizer_rgb.step()
            optimizer_depth.step()

        loss_rgb_list.append(loss_rgb.item())
        loss_depth_list.append(loss_depth.item())

        if batch_idx % 10 == 0:
            pbar.set_postfix({
                'RGB Loss': f'{loss_rgb.item():.4f}',
                'Depth Loss': f'{loss_depth.item():.4f}',
                'Avg RGB': f'{np.mean(loss_rgb_list):.4f}',
                'Avg Depth': f'{np.mean(loss_depth_list):.4f}'
            })

        if batch_idx % 50 == 0:
            msg = f"Epoch [{epoch}] Batch [{batch_idx}/{len(train_loader)}] " \
                  f"Loss_RGB: {loss_rgb.item():.4f} Loss_Depth: {loss_depth.item():.4f}"
            with open(log_file, "a") as f:
                f.write(msg + "\n")

    avg_loss_rgb = np.mean(loss_rgb_list)
    avg_loss_depth = np.mean(loss_depth_list)
    return avg_loss_rgb, avg_loss_depth


def compute_val_loss(teacher_rgb, teacher_depth, student_rgb, student_depth,
                     val_loader, device, amp_dtype=None, channels_last=False):
    """计算验证集上的 RGB 蒸馏损失（用于早停判断）

    只算 RGB 分支：depth loss 与 rgb loss 高度相关，省掉 depth 学生的前向
    """
    student_rgb.eval()

    use_amp = amp_dtype is not None
    loss_rgb_list = []

    with torch.no_grad():
        for data in val_loader:
            rgb = data["intensity"].to(device, non_blocking=True)
            depth = data["depth"].to(device, non_blocking=True)
            if channels_last:
                rgb = rgb.contiguous(memory_format=torch.channels_last)
                depth = depth.contiguous(memory_format=torch.channels_last)

            with (autocast(enabled=use_amp, dtype=amp_dtype) if use_amp else nullcontext()):
                feat_t_rgb = teacher_rgb(rgb)
                feat_t_depth = teacher_depth(depth)

                proj_d, proj_d_amply, feat_s_rgb, feat_s_rgb_am = student_rgb(feat_t_rgb, feat_t_depth)
                loss_rgb = (
                    loss_distil(feat_s_rgb, feat_t_rgb) +
                    loss_distil(proj_d, feat_t_rgb) +
                    loss_distil(proj_d_amply, feat_t_rgb) +
                    loss_distil(feat_s_rgb_am, feat_t_rgb)
                )
                del proj_d, proj_d_amply, feat_s_rgb, feat_s_rgb_am

            loss_rgb_list.append(loss_rgb.item())

    avg_loss_rgb = np.mean(loss_rgb_list)
    # total 等同于 rgb loss（保持兼容）
    return avg_loss_rgb, 0.0, avg_loss_rgb


def evaluate(teacher_rgb, teacher_depth, student_rgb, student_depth,
             test_loader, device, log_file, amp_dtype=None, channels_last=False):
    """评估测试集，计算图像级 AUROC（patch 分数聚合到原图后取 max）"""
    student_rgb.eval()
    student_depth.eval()

    use_amp = amp_dtype is not None

    # 收集每个 (frame_id) 的所有 patch 分数和标签
    img_scores: Dict[str, list] = {}
    img_labels: Dict[str, int] = {}

    with torch.no_grad():
        for data in test_loader:
            rgb = data["intensity"].to(device, non_blocking=True)
            depth = data["depth"].to(device, non_blocking=True)
            if channels_last:
                rgb = rgb.contiguous(memory_format=torch.channels_last)
                depth = depth.contiguous(memory_format=torch.channels_last)
            labels = data["label"].cpu().numpy()
            frame_ids = data["frame_id"]  # list of strings

            with (autocast(enabled=use_amp, dtype=amp_dtype) if use_amp else nullcontext()):
                feat_t_rgb = teacher_rgb(rgb)
                feat_t_depth = teacher_depth(depth)
                proj_d, proj_d_amply, feat_s_rgb, feat_s_rgb_am = student_rgb(feat_t_rgb, feat_t_depth)
                proj_r, proj_r_amply, feat_s_depth, feat_s_depth_am = student_depth(feat_t_depth, feat_t_rgb)

            amap_rgb, _ = cal_anomaly_map(feat_s_rgb, feat_t_rgb, out_size=(256, 256), amap_mode='mul')
            amap_depth, _ = cal_anomaly_map(feat_s_depth, feat_t_depth, out_size=(256, 256), amap_mode='mul')

            amap = amap_rgb + amap_depth

            if amap.ndim == 3:
                scores = amap.reshape(amap.shape[0], -1).max(axis=1)
            else:
                scores = np.array([amap.max()])

            # 按 frame_id 分组（同一原图的不同 patch 归到同一个 key）
            for score, label, fid in zip(scores, labels, frame_ids):
                if fid not in img_scores:
                    img_scores[fid] = []
                    img_labels[fid] = int(label)
                img_scores[fid].append(float(score))

    # 图像级：取每张图所有 patch 的最大分数
    image_score_list = []
    image_label_list = []
    for fid in sorted(img_scores.keys()):
        max_score = max(img_scores[fid])
        image_score_list.append(max_score)
        image_label_list.append(img_labels[fid])

    anomaly_scores = np.array(image_score_list)
    labels_arr = np.array(image_label_list)

    if len(np.unique(labels_arr)) < 2:
        msg = "Warning: Only one class in test set, cannot compute AUROC"
        print(msg)
        with open(log_file, "a") as f:
            f.write(msg + "\n")
        return 0.0

    auroc = roc_auc_score(labels_arr, anomaly_scores)
    return auroc


def main(args):
    # 设置随机种子
    setup_seed(args.seed)

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建带时间戳和配置信息的输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_str = f"cam{args.view_id}_bs{args.batch_size}_lr{args.lr}_img{args.img_size}_ratio{args.train_sample_ratio}"
    run_dir = os.path.join(args.save_dir, f"{timestamp}_{config_str}")
    os.makedirs(run_dir, exist_ok=True)

    # 日志文件路径
    log_file = os.path.join(run_dir, "training.log")

    print(f"Results will be saved to: {run_dir}")

    with open(log_file, "w") as f:
        f.write(f"Training started at {timestamp}\n")
        f.write(f"Args: {args}\n\n")

    # 创建数据集
    print(f"Loading dataset from {args.train_root} and {args.test_root}")
    train_dataset = RailDualModalDataset(
        train_root=args.train_root,
        test_root=args.test_root,
        view_id=args.view_id,
        split="train",
        img_size=args.img_size,
        depth_norm=args.depth_norm,
        use_patch=args.use_patch,            # 训练默认 False（整图 resize，速度优先）
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        train_sample_ratio=args.train_sample_ratio,
        train_sample_num=args.train_sample_num,
        random_seed=args.random_seed_sample,
        preload=args.preload,
        preload_workers=args.preload_workers,
        train_val_test_split=args.train_val_test_split,
        sampling_mode=args.sampling_mode,    # 时间均匀采样保证训练分布稳定
    )

    val_dataset = RailDualModalDataset(
        train_root=args.train_root,
        test_root=args.test_root,
        view_id=args.view_id,
        split="val",
        img_size=args.img_size,
        depth_norm=args.depth_norm,
        use_patch=args.use_patch,
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        train_sample_ratio=args.train_sample_ratio,
        train_sample_num=args.train_sample_num,
        random_seed=args.random_seed_sample,
        preload=args.preload,
        preload_workers=args.preload_workers,
        train_val_test_split=args.train_val_test_split,
        sampling_mode=args.sampling_mode,
    )

    # 测试集独立配置 patch（保持原始空间分辨率，便于检测小缺陷）
    test_dataset = RailDualModalDataset(
        train_root=args.train_root,
        test_root=args.test_root,
        view_id=args.view_id,
        split="test",
        img_size=args.img_size,
        depth_norm=args.depth_norm,
        use_patch=args.test_use_patch,           # 测试默认 True
        patch_size=args.test_patch_size,
        patch_stride=args.test_patch_stride,
        preload=args.preload,
        preload_workers=args.preload_workers,
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")

    # DataLoader 关键加速：
    # - persistent_workers: worker 进程跨 epoch 复用，避免反复初始化
    # - prefetch_factor: 每个 worker 预取 4 个 batch，掩盖 IO 抖动
    nw = args.num_workers
    common_loader_kwargs = dict(
        num_workers=nw,
        pin_memory=True,
        persistent_workers=(nw > 0),
        prefetch_factor=(4 if nw > 0 else None),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,                  # 与 channels_last + AMP 共用，避免最后小 batch 触发重编译
        **common_loader_kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **common_loader_kwargs,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **common_loader_kwargs,
    )

    # 创建模型
    print("Creating models...")
    teacher_rgb = ResNet50Encoder(pretrained=True).to(device)
    teacher_depth = ResNet50Encoder(pretrained=True).to(device)

    # 冻结教师网络
    teacher_rgb.eval()
    teacher_depth.eval()
    for param in teacher_rgb.parameters():
        param.requires_grad = False
    for param in teacher_depth.parameters():
        param.requires_grad = False

    # 创建学生网络
    student_rgb = ResNet50DualModalDecoder(
        pretrained=False
    ).to(device)

    student_depth = ResNet50DualModalDecoder(
        pretrained=False
    ).to(device)

    # channels_last 内存格式（NHWC）：在 Tensor Core GPU 上 ConvNet 提速 10-30%
    if args.channels_last and device.type == "cuda":
        teacher_rgb = teacher_rgb.to(memory_format=torch.channels_last)
        teacher_depth = teacher_depth.to(memory_format=torch.channels_last)
        student_rgb = student_rgb.to(memory_format=torch.channels_last)
        student_depth = student_depth.to(memory_format=torch.channels_last)
        print("Memory format: channels_last (NHWC)")

    # 优化器
    optimizer_rgb = torch.optim.Adam(
        student_rgb.parameters(),
        lr=args.lr,
        betas=(0.5, 0.999)
    )
    optimizer_depth = torch.optim.Adam(
        student_depth.parameters(),
        lr=args.lr,
        betas=(0.5, 0.999)
    )

    # 学习率调度器
    scheduler_rgb = torch.optim.lr_scheduler.ExponentialLR(optimizer_rgb, gamma=0.95)
    scheduler_depth = torch.optim.lr_scheduler.ExponentialLR(optimizer_depth, gamma=0.95)

    # 混合精度配置：
    #   - bf16: RTX 6000 Pro / Ampere+ 原生支持，范围与 fp32 相同，无需 GradScaler，训练最稳
    #   - fp16: 老 GPU(Turing) 备选，需 GradScaler 防溢出
    #   - fp32: 关闭 AMP（兼容性兜底）
    amp_dtype = None
    use_scaler = False
    if args.precision == "bf16" and AMP_AVAILABLE and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
        print("Mixed precision: bfloat16 (no GradScaler needed)")
    elif args.precision == "fp16" and AMP_AVAILABLE:
        amp_dtype = torch.float16
        use_scaler = True
        print("Mixed precision: fp16 (GradScaler enabled)")
    elif args.precision == "bf16":
        # 用户要 bf16 但硬件不支持 → 自动降级 fp16
        amp_dtype = torch.float16
        use_scaler = True
        print("bf16 not supported on this GPU, falling back to fp16")
    else:
        print("Precision: fp32 (AMP disabled)")

    scaler = make_scaler(enabled=use_scaler) if use_scaler else None

    # 保存原始模型引用（compile 失败时回退用）
    teacher_rgb_org, teacher_depth_org = teacher_rgb, teacher_depth
    student_rgb_org, student_depth_org = student_rgb, student_depth

    # torch.compile：图级别 JIT 融合 kernel，减少 Python 开销（PyTorch >= 2.0）
    # 部分环境缺少 libcuda.so / Triton 等依赖，warmup 验证编译是否可用
    if args.compile and hasattr(torch, 'compile'):
        print("Applying torch.compile (reduce-overhead mode)...")
        teacher_rgb_compiled = torch.compile(teacher_rgb, mode="reduce-overhead")
        teacher_depth_compiled = torch.compile(teacher_depth, mode="reduce-overhead")
        try:
            student_rgb_compiled = torch.compile(student_rgb, mode="reduce-overhead")
            student_depth_compiled = torch.compile(student_depth, mode="reduce-overhead")
        except Exception as e:
            print(f"  Student compile failed ({e}), compiling teachers only")
            student_rgb_compiled = student_rgb
            student_depth_compiled = student_depth

        # Warmup：JIT 编译在首次 forward 触发，这里提前验证避免训练中途崩溃
        print("  Warming up compiled models...")
        try:
            dummy_rgb = torch.randn(1, 3, 256, 256, device=device)
            dummy_depth = torch.randn(1, 3, 256, 256, device=device)
            with torch.no_grad():
                _ = teacher_rgb_compiled(dummy_rgb)
                _ = teacher_depth_compiled(dummy_depth)
                feat_t = teacher_rgb_compiled(dummy_rgb)
                feat_d = teacher_depth_compiled(dummy_depth)
                _ = student_rgb_compiled(feat_t, feat_d)
                _ = student_depth_compiled(feat_d, feat_t)
            teacher_rgb, teacher_depth = teacher_rgb_compiled, teacher_depth_compiled
            student_rgb, student_depth = student_rgb_compiled, student_depth_compiled
            print("  torch.compile OK")
        except Exception as e:
            print(f"  torch.compile warmup failed ({type(e).__name__}), falling back to eager mode")
            del teacher_rgb_compiled, teacher_depth_compiled
            del student_rgb_compiled, student_depth_compiled
            teacher_rgb, teacher_depth = teacher_rgb_org, teacher_depth_org
            student_rgb, student_depth = student_rgb_org, student_depth_org
            args.compile = False
    elif args.compile:
        print("torch.compile requested but not available (need PyTorch >= 2.0)")

    # 训练循环
    # 使用验证损失而非 AUROC 做早停（验证集仅有正常样本，无法计算 AUROC）
    best_val_loss = float('inf')
    best_epoch_saved = 0
    patience = 15
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        msg = f"\n{'='*50}\nEpoch {epoch}/{args.epochs}\n{'='*50}"
        print(msg)
        with open(log_file, "a") as f:
            f.write(msg + "\n")

        # 训练
        avg_loss_rgb, avg_loss_depth = train_one_epoch(
            teacher_rgb, teacher_depth, student_rgb, student_depth,
            train_loader, optimizer_rgb, optimizer_depth,
            device, epoch, log_file,
            scaler=scaler, amp_dtype=amp_dtype, channels_last=args.channels_last,
        )

        # 更新学习率
        scheduler_rgb.step()
        scheduler_depth.step()

        msg = f"Epoch {epoch} - Avg Loss RGB: {avg_loss_rgb:.4f}, Avg Loss Depth: {avg_loss_depth:.4f}"
        print(msg)
        with open(log_file, "a") as f:
            f.write(msg + "\n")

        # 在验证集上评估（用蒸馏损失做早停）
        if epoch % args.eval_interval == 0:
            val_loss_rgb, val_loss_depth, val_loss_total = compute_val_loss(
                teacher_rgb, teacher_depth, student_rgb, student_depth,
                val_loader, device,
                amp_dtype=amp_dtype, channels_last=args.channels_last,
            )

            msg = f"Epoch {epoch} - Val Loss RGB: {val_loss_rgb:.4f}, Val Loss Depth: {val_loss_depth:.4f}, Total: {val_loss_total:.4f}"
            print(msg)
            with open(log_file, "a") as f:
                f.write(msg + "\n")

            # 保存最佳模型（基于验证总损失，越低越好）
            if val_loss_total < best_val_loss:
                best_val_loss = val_loss_total
                best_epoch_saved = epoch
                patience_counter = 0
                ckpt_path = os.path.join(run_dir, f"best_cam{args.view_id}.pth")
                # 注意：best_val_loss 必须 cast 成 Python float
                # 否则 numpy.float64 会触发 PyTorch 2.6 weights_only=True 报错
                torch.save({
                    'epoch': int(epoch),
                    'student_rgb': student_rgb.state_dict(),
                    'student_depth': student_depth.state_dict(),
                    'best_val_loss': float(best_val_loss),
                }, ckpt_path)
                msg = f"Saved best model to {ckpt_path}"
                print(msg)
                with open(log_file, "a") as f:
                    f.write(msg + "\n")
            else:
                patience_counter += 1
                msg = f"No improvement for {patience_counter} evaluation(s)"
                print(msg)
                with open(log_file, "a") as f:
                    f.write(msg + "\n")

            # 提前停止
            if patience_counter >= patience:
                msg = f"Early stopping at epoch {epoch} (no improvement for {patience} evaluations)"
                print(msg)
                with open(log_file, "a") as f:
                    f.write(msg + "\n")
                break

    msg = f"\nTraining finished! Best Val Loss: {best_val_loss:.4f}"
    print(msg)
    with open(log_file, "a") as f:
        f.write(msg + "\n")

    # 加载最佳模型并在测试集上做最终评估
    best_ckpt_path = os.path.join(run_dir, f"best_cam{args.view_id}.pth")
    print("\n" + "=" * 50)
    print("Final evaluation on test set (best checkpoint)...")
    print("=" * 50)

    # 加载 best ckpt（保证最终评估用的是验证集最佳模型，而不是最后一轮）
    if os.path.exists(best_ckpt_path):
        # PyTorch 2.6+ 默认 weights_only=True，会拒绝 numpy 标量等非张量对象。
        # 当前 ckpt 由本脚本生成，来源可信 → 显式 weights_only=False。
        # 同时兼容 PyTorch < 2.4 不支持该参数的情况。
        try:
            ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        except TypeError:
            # 老版本 torch 没有 weights_only 参数
            ckpt = torch.load(best_ckpt_path, map_location=device)
        # compile 后的模型 state_dict key 会带 _orig_mod. 前缀，兼容加载
        def _strip_prefix(sd):
            return {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        try:
            student_rgb.load_state_dict(_strip_prefix(ckpt['student_rgb']))
            student_depth.load_state_dict(_strip_prefix(ckpt['student_depth']))
            print(f"Loaded best checkpoint from epoch {ckpt.get('epoch', '?')}")
        except Exception as e:
            print(f"Failed to load best ckpt ({e}), evaluating with current weights")

    test_auroc = evaluate(
        teacher_rgb, teacher_depth, student_rgb, student_depth,
        test_loader, device, log_file,
        amp_dtype=amp_dtype, channels_last=args.channels_last,
    )

    msg = f"\n{'='*50}\n"
    msg += f"Final Result - Cam {args.view_id}\n"
    msg += f"  Best epoch: {best_epoch_saved}\n"
    msg += f"  Best Val Loss: {best_val_loss:.4f}\n"
    msg += f"  Test AUROC: {test_auroc:.4f}\n"
    msg += f"  Model: {best_ckpt_path}\n"
    msg += f"{'='*50}"
    print(msg)
    with open(log_file, "a") as f:
        f.write(msg + "\n")

    # ============================================================
    # 训练结束清理：仅保留 best 模型，删除其它 .pth / .pt 检查点
    # 节省磁盘空间（特别是 8 视角并发训练时）
    # ============================================================
    try:
        kept = os.path.basename(best_ckpt_path)
        removed = []
        for fname in os.listdir(run_dir):
            if fname.endswith((".pth", ".pt", ".ckpt")) and fname != kept:
                fpath = os.path.join(run_dir, fname)
                os.remove(fpath)
                removed.append(fname)
        if removed:
            print(f"Cleaned up {len(removed)} non-best checkpoints: {removed}")
            with open(log_file, "a") as f:
                f.write(f"Cleaned up: {removed}\n")
    except Exception as e:
        print(f"Checkpoint cleanup warning: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TRD on Rail Dataset")

    # 数据参数
    parser.add_argument("--train_root", type=str, default="/data1/Leaddo_data/20260327-resize512",
                        help="Path to training dataset (推荐用已 resize 的目录，如 20260327-resize512)")
    parser.add_argument("--test_root", type=str, default="/home/root123/LF/WYM/SteelRailWay/rail_mvtec_gt_test",
                        help="Path to test dataset")
    parser.add_argument("--view_id", type=int, default=1,
                        help="Camera view ID (1-8 for train, 1-6 for test)")
    parser.add_argument("--img_size", type=int, default=512,
                        help="Input image size (与 Railway 参考实现对齐)")
    parser.add_argument("--depth_norm", type=str, default="zscore",
                        choices=["zscore", "minmax", "log"],
                        help="Depth normalization method")

    # ============== 训练阶段 patch ==============
    # 默认关闭：整图 resize→512，比 patch 切分快 7 倍（论文默认也是整图）
    parser.add_argument("--use_patch", action="store_true", default=False,
                        help="(训练) 是否使用 patch 切分，默认 False = 整图 resize")
    parser.add_argument("--patch_size", type=int, default=900,
                        help="(训练) Patch size when use_patch=True")
    parser.add_argument("--patch_stride", type=int, default=850,
                        help="(训练) Patch stride")

    # ============== 测试阶段 patch ==============
    # 默认开启：测试集原图较大，保持 patch 推理 + max 聚合 → 检测小缺陷
    parser.add_argument("--test_use_patch", action="store_true", default=True,
                        help="(测试) 是否使用 patch 切分，默认 True 提升小缺陷召回")
    parser.add_argument("--no_test_use_patch", action="store_false", dest="test_use_patch")
    parser.add_argument("--test_patch_size", type=int, default=900,
                        help="(测试) Patch size")
    parser.add_argument("--test_patch_stride", type=int, default=850,
                        help="(测试) Patch stride")

    # 数据采样参数
    parser.add_argument("--train_sample_ratio", type=float, default=1.0,
                        help="Training data sampling ratio (0-1, 默认 1.0 用满全部数据)")
    parser.add_argument("--train_sample_num", type=int, default=None,
                        help="Training data sampling number (overrides ratio if set)")
    parser.add_argument("--random_seed_sample", type=int, default=42,
                        help="Random seed for data sampling")
    parser.add_argument("--train_val_test_split", type=float, nargs=3, default=[0.9, 0.1, 0.0],
                        help="Train/Val/Test split ratio (test 来自 rail_mvtec_gt_test，所以这里 test=0)")
    parser.add_argument("--sampling_mode", type=str, default="uniform_time",
                        choices=["random", "uniform_time"],
                        help="采样模式：random=随机采样；uniform_time=按文件名(时间)均匀采样")

    # 数据预加载参数
    parser.add_argument("--preload", action="store_true",
                        help="Preload all images to memory (强烈推荐)")
    parser.add_argument("--preload_workers", type=int, default=16,
                        help="Number of workers for preloading (default 16)")

    # 训练参数（针对 RTX 6000 Pro 80GB 调优）
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size (RTX 6000 Pro 80GB 可上 128~192)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="Learning rate")
    parser.add_argument("--eval_interval", type=int, default=5,
                        help="Evaluation interval")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of dataloader workers")

    # 其他参数
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use")
    parser.add_argument("--save_dir", type=str, default="./outputs/rail",
                        help="Directory to save checkpoints and logs")

    # 精度选择（bf16 在 Ampere/Ada/Blackwell 是首选）
    parser.add_argument("--precision", type=str, default="bf16",
                        choices=["bf16", "fp16", "fp32"],
                        help="训练精度：bf16 (推荐 RTX 6000 Pro)、fp16、fp32")
    parser.add_argument("--channels_last", action="store_true", default=True,
                        help="使用 channels_last (NHWC) 内存格式，Tensor Core 提速")
    parser.add_argument("--no_channels_last", action="store_false", dest="channels_last")

    parser.add_argument("--compile", action="store_true", default=True,
                        help="Enable torch.compile graph JIT (PyTorch >= 2.0)")
    parser.add_argument("--no_compile", action="store_false", dest="compile",
                        help="Disable torch.compile")

    args = parser.parse_args()
    main(args)
