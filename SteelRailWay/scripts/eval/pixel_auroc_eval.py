#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pixel-level AUROC evaluation script for the railway TRD (Teacher-Reverse-Distillation) model.

Unlike the existing image-level evaluation (cam4_branch_auroc_test.py) which takes per-patch
max anomaly scores, this script collects per-patch anomaly maps alongside pixel-level GT
masks, flattens all pixels, and computes pixel AUROC via sklearn.metrics.roc_auc_score.

This is the same evaluation pattern used by the MVTec 3D-AD pipeline
(train_trd_mvtec3d_rgbd.py:test_metric).

Usage:
    python test/pixel_auroc_eval.py \
        --ckpt outputs/rail_all/Cam4/xxx/best_cam4.pth \
        --train_root /data1/Leaddo_data/20260327-resize512 \
        --test_root ./rail_mvtec_gt_test \
        --view_id 4
"""

import sys
import argparse
from pathlib import Path
from contextlib import nullcontext
from collections import defaultdict


# ---------------------------------------------------------------------------
#  Utility helpers (consistent with cam4_branch_auroc_test.py)
# ---------------------------------------------------------------------------

def bootstrap_project(project_root):
    project_root = Path(project_root).resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def safe_load_ckpt(path, device):
    import torch
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def strip_prefix(sd):
    return {k.replace("_orig_mod.", ""): v for k, v in sd.items()}


def normalize_depth_peft_state(state_dict):
    """Keep only the DepthAffinePEFT gain/bias tensors from several checkpoint layouts."""
    normalized = {}
    for key, value in state_dict.items():
        clean = (
            key.replace("_orig_mod.", "")
            .replace("module.", "")
            .replace("peft.", "")
            .replace("depth_peft.", "")
        )
        if clean in {"gain", "bias"}:
            normalized[clean] = value
    return normalized


def load_depth_peft(path, device, peft_cls):
    payload = safe_load_ckpt(path, device)
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
        metadata = {k: v for k, v in payload.items() if k != "state_dict"}
    else:
        state_dict = payload
        metadata = {}

    if not isinstance(state_dict, dict):
        raise TypeError(f"Invalid depth PEFT checkpoint: {path}")

    peft = peft_cls().to(device)
    peft.load_state_dict(normalize_depth_peft_state(state_dict), strict=True)
    peft.eval()
    for param in peft.parameters():
        param.requires_grad_(False)
    return peft, metadata


def load_depth_peft_from_joint_ckpt(payload, device, peft_cls):
    if not isinstance(payload, dict):
        return None, {}
    if "depth_peft" in payload and isinstance(payload["depth_peft"], dict):
        state_dict = payload["depth_peft"]
        metadata = {"source": "joint_ckpt.depth_peft"}
    else:
        state_dict = normalize_depth_peft_state(payload)
        metadata = {"source": "joint_ckpt.flat_keys"} if state_dict else {}
    if not state_dict:
        return None, {}

    peft = peft_cls().to(device)
    peft.load_state_dict(normalize_depth_peft_state(state_dict), strict=True)
    peft.eval()
    for param in peft.parameters():
        param.requires_grad_(False)
    return peft, metadata


def find_cam_ckpt(runs_root, view_id):
    root = Path(runs_root)
    candidates = list(root.glob(f"Cam{view_id}/**/best_cam{view_id}.pth"))
    if not candidates:
        candidates = list(root.glob(f"**/best_cam{view_id}.pth"))
    if not candidates:
        raise FileNotFoundError(
            f"Cannot find best_cam{view_id}.pth under: {runs_root}"
        )
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0])


def resolve_amp_dtype(precision, device):
    import torch
    if device.type != "cuda":
        return None
    if precision == "bf16" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if precision in {"bf16", "fp16"}:
        return torch.float16
    return None


def cal_anomaly_map_fp32(fs_list, ft_list, out_size=(256, 256), amap_mode="mul"):
    """Standalone anomaly map computation that forces fp32 to avoid bf16 -> numpy errors."""
    import torch
    import torch.nn.functional as F

    if isinstance(out_size, int):
        out_size = (out_size, out_size)

    b = fs_list[0].shape[0]
    if amap_mode == "mul":
        anomaly_map = torch.ones((b, *out_size), device=fs_list[0].device)
    else:
        anomaly_map = torch.zeros((b, *out_size), device=fs_list[0].device)

    for fs, ft in zip(fs_list, ft_list):
        fs = fs.float()
        ft = ft.float()
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = a_map.unsqueeze(1)
        a_map = F.interpolate(a_map, size=out_size, mode="bilinear", align_corners=True)
        a_map = a_map[:, 0, :, :]

        if amap_mode == "mul":
            anomaly_map *= a_map
        else:
            anomaly_map += a_map

    return anomaly_map.float().cpu().numpy()


# ---------------------------------------------------------------------------
#  Main evaluation
# ---------------------------------------------------------------------------

def main():
    _default_proj = str(Path(__file__).resolve().parents[2])
    parser = argparse.ArgumentParser(
        description="Pixel-level AUROC evaluation for railway TRD model"
    )
    parser.add_argument("--project_root", default=_default_proj)
    parser.add_argument(
        "--train_root", default=str(Path(_default_proj) / "data_20260327")
    )
    parser.add_argument("--test_root", default=str(Path(_default_proj) / "rail_mvtec_gt_test"))
    parser.add_argument("--runs_root", default=str(Path(_default_proj) / "outputs/rail_all"))
    parser.add_argument("--ckpt", default=None, help="Path to checkpoint; auto-find if not given")
    parser.add_argument(
        "--depth_peft_ckpt",
        default=None,
        help="Optional DepthAffinePEFT checkpoint applied to the depth teacher.",
    )
    parser.add_argument(
        "--out_csv",
        default=str(Path(_default_proj) / "outputs/pixel_auroc/pixel_auroc_results.csv"),
    )

    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--precision", choices=["bf16", "fp16", "fp32"], default="bf16"
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--depth_norm", default="zscore")
    parser.add_argument("--view_id", type=int, default=1)
    parser.add_argument("--patch_size", type=int, default=900)
    parser.add_argument("--patch_stride", type=int, default=850)
    parser.add_argument(
        "--module_ablation",
        choices=["full", "no_cf", "no_ca", "no_cf_ca"],
        default="full",
        help="Decoder module path used at inference, matching eval_from_ckpt.py.",
    )
    parser.add_argument("--channels_last", action="store_true", default=True)
    parser.add_argument("--no_channels_last", action="store_false", dest="channels_last")
    parser.add_argument(
        "--smooth_sigma",
        type=float,
        default=4.0,
        help="Gaussian filter sigma for anomaly map smoothing (0 = disable)",
    )
    args = parser.parse_args()

    import cv2
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from sklearn.metrics import roc_auc_score
    from scipy.ndimage import gaussian_filter

    bootstrap_project(args.project_root)

    from datasets.rail_dataset import RailDualModalDataset
    from models.trd.encoder import ResNet50Encoder
    from models.trd.decoder import ResNet50DualModalDecoder
    from rail_peft import DepthAffinePEFT, DepthEncoderWithPEFT

    ckpt_path = args.ckpt or find_cam_ckpt(args.runs_root, args.view_id)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    amp_dtype = resolve_amp_dtype(args.precision, device)

    print(f"Device: {device}")
    print(f"Precision: {args.precision if amp_dtype is not None else 'fp32'}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Depth PEFT: {args.depth_peft_ckpt or 'N/A'}")
    print(f"Module ablation: {args.module_ablation}")
    print(f"Gaussian sigma: {args.smooth_sigma}")

    # ---- Dataset & DataLoader ----
    dataset = RailDualModalDataset(
        train_root=args.train_root,
        test_root=args.test_root,
        view_id=args.view_id,
        split="test",
        img_size=args.img_size,
        depth_norm=args.depth_norm,
        use_patch=True,
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        preload=False,
    )

    loader_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    loader = DataLoader(dataset, **loader_kwargs)

    # ---- Models ----
    teacher_rgb = ResNet50Encoder(pretrained=True).to(device).eval()
    teacher_depth = ResNet50Encoder(pretrained=True).to(device).eval()
    student_rgb = ResNet50DualModalDecoder(
        pretrained=False,
        module_ablation=args.module_ablation,
    ).to(device).eval()
    student_depth = ResNet50DualModalDecoder(
        pretrained=False,
        module_ablation=args.module_ablation,
    ).to(device).eval()

    for p in teacher_rgb.parameters():
        p.requires_grad = False
    for p in teacher_depth.parameters():
        p.requires_grad = False

    if args.channels_last and device.type == "cuda":
        teacher_rgb = teacher_rgb.to(memory_format=torch.channels_last)
        teacher_depth = teacher_depth.to(memory_format=torch.channels_last)
        student_rgb = student_rgb.to(memory_format=torch.channels_last)
        student_depth = student_depth.to(memory_format=torch.channels_last)

    ckpt = safe_load_ckpt(ckpt_path, device)
    student_rgb.load_state_dict(strip_prefix(ckpt["student_rgb"]))
    student_depth.load_state_dict(strip_prefix(ckpt["student_depth"]))

    if args.depth_peft_ckpt:
        peft, _ = load_depth_peft(args.depth_peft_ckpt, device, DepthAffinePEFT)
        teacher_depth = DepthEncoderWithPEFT(teacher_depth, peft).to(device).eval()
        print(
            "Loaded Depth PEFT: "
            f"gain={float(peft.gain.detach().cpu()):.6f}, "
            f"bias={float(peft.bias.detach().cpu()):.6f}"
        )
    else:
        joint_peft, _ = load_depth_peft_from_joint_ckpt(ckpt, device, DepthAffinePEFT)
        if joint_peft is not None:
            teacher_depth = DepthEncoderWithPEFT(teacher_depth, joint_peft).to(device).eval()
            print(
                "Loaded embedded Depth PEFT: "
                f"gain={float(joint_peft.gain.detach().cpu()):.6f}, "
                f"bias={float(joint_peft.bias.detach().cpu()):.6f}"
            )

    print(
        f"Loaded epoch={ckpt.get('epoch', 'N/A')}, "
        f"best_val_loss={ckpt.get('best_val_loss', 'N/A')}"
    )
    print(f"Test patches: {len(dataset)}")

    # ---- Pixel-level accumulation ----
    gt_list_px = []       # per-patch pixel GT arrays
    pr_list_px_fused = [] # per-patch pixel anomaly score arrays (fused)
    pr_list_px_rgb = []   # per-patch pixel anomaly score arrays (rgb)
    pr_list_px_depth = [] # per-patch pixel anomaly score arrays (depth)

    # Image-level (for reference)
    img_labels: dict = {}
    img_scores_fused: dict = defaultdict(list)
    img_scores_rgb: dict = defaultdict(list)
    img_scores_depth: dict = defaultdict(list)

    gt_size = (args.img_size, args.img_size)

    for data in loader:
        rgb = data["intensity"].to(device, non_blocking=True)
        depth = data["depth"].to(device, non_blocking=True)

        if args.channels_last and device.type == "cuda":
            rgb = rgb.contiguous(memory_format=torch.channels_last)
            depth = depth.contiguous(memory_format=torch.channels_last)

        gts = data["gt"].cpu().numpy()          # [B, H, W] binary
        labels = data["label"].cpu().numpy()
        frame_ids = data["frame_id"]

        amp_ctx = (
            torch.amp.autocast(device_type=device.type, dtype=amp_dtype)
            if amp_dtype is not None
            else nullcontext()
        )

        with torch.no_grad(), amp_ctx:
            feat_t_rgb = teacher_rgb(rgb)
            feat_t_depth = teacher_depth(depth)

            _, _, feat_s_rgb, _ = student_rgb(feat_t_rgb, feat_t_depth)
            _, _, feat_s_depth, _ = student_depth(feat_t_depth, feat_t_rgb)

        amap_rgb = cal_anomaly_map_fp32(
            feat_s_rgb, feat_t_rgb, out_size=(256, 256), amap_mode="mul"
        )
        amap_depth = cal_anomaly_map_fp32(
            feat_s_depth, feat_t_depth, out_size=(256, 256), amap_mode="mul"
        )
        amap_fused = amap_rgb + amap_depth

        # ---- Per-patch pixel accumulation ----
        for b in range(amap_fused.shape[0]):
            amap_rgb_b = amap_rgb[b]        # [256, 256]
            amap_depth_b = amap_depth[b]    # [256, 256]
            amap_fused_b = amap_fused[b]    # [256, 256]
            gt_b = gts[b]                    # [H, W] at img_size

            # Gaussian smoothing (on the 256x256 anomaly maps, then upscale)
            if args.smooth_sigma > 0:
                amap_rgb_b = gaussian_filter(amap_rgb_b, sigma=args.smooth_sigma)
                amap_depth_b = gaussian_filter(amap_depth_b, sigma=args.smooth_sigma)
                amap_fused_b = gaussian_filter(amap_fused_b, sigma=args.smooth_sigma)

            # Upscale anomaly maps to match GT resolution
            amap_rgb_up = cv2.resize(
                amap_rgb_b, gt_size, interpolation=cv2.INTER_LINEAR
            )
            amap_depth_up = cv2.resize(
                amap_depth_b, gt_size, interpolation=cv2.INTER_LINEAR
            )
            amap_fused_up = cv2.resize(
                amap_fused_b, gt_size, interpolation=cv2.INTER_LINEAR
            )

            # Flatten and accumulate
            gt_list_px.append(gt_b.ravel().astype(np.uint8, copy=False))
            pr_list_px_rgb.append(amap_rgb_up.ravel().astype(np.float32, copy=False))
            pr_list_px_depth.append(amap_depth_up.ravel().astype(np.float32, copy=False))
            pr_list_px_fused.append(amap_fused_up.ravel().astype(np.float32, copy=False))

        # Image-level scores (max aggregation, for reference)
        rgb_img_scores = amap_rgb.reshape(amap_rgb.shape[0], -1).max(axis=1)
        depth_img_scores = amap_depth.reshape(amap_depth.shape[0], -1).max(axis=1)
        fused_img_scores = amap_fused.reshape(amap_fused.shape[0], -1).max(axis=1)

        for fid, label, sr, sd, sf in zip(
            frame_ids, labels, rgb_img_scores, depth_img_scores, fused_img_scores
        ):
            img_labels[str(fid)] = int(label)
            img_scores_rgb[str(fid)].append(float(sr))
            img_scores_depth[str(fid)].append(float(sd))
            img_scores_fused[str(fid)].append(float(sf))

    # ---- Compute metrics ----
    gt_arr = (
        np.concatenate(gt_list_px).astype(np.int32, copy=False)
        if gt_list_px else np.array([], dtype=np.int32)
    )

    def safe_pixel_auroc(name, pr_list):
        arr = (
            np.concatenate(pr_list).astype(np.float64, copy=False)
            if pr_list else np.array([], dtype=np.float64)
        )
        if len(np.unique(gt_arr)) < 2:
            return None
        return float(roc_auc_score(gt_arr, arr))

    auroc_px_fused = safe_pixel_auroc("fused", pr_list_px_fused)
    auroc_px_rgb = safe_pixel_auroc("rgb", pr_list_px_rgb)
    auroc_px_depth = safe_pixel_auroc("depth", pr_list_px_depth)

    # Image-level AUROC (reference)
    def image_auroc(scores_dict):
        fids = sorted(img_labels.keys())
        labels_arr = np.array([img_labels[f] for f in fids])
        scores_arr = np.array([max(scores_dict[f]) for f in fids])
        if len(np.unique(labels_arr)) < 2:
            return None
        return float(roc_auc_score(labels_arr, scores_arr))

    auroc_img_fused = image_auroc(img_scores_fused)
    auroc_img_rgb = image_auroc(img_scores_rgb)
    auroc_img_depth = image_auroc(img_scores_depth)

    # ---- Results ----
    def fmt(x):
        return "N/A" if x is None else f"{x:.4f}"

    n_images = len(img_labels)
    labels_arr = np.array(list(img_labels.values()))
    n_abnormal = int((labels_arr == 1).sum())
    n_normal = int((labels_arr == 0).sum())
    n_pixels = int(gt_arr.size)

    print()
    print("=" * 60)
    print("Pixel AUROC Evaluation")
    print("=" * 60)
    print(f"Checkpoint     : {ckpt_path}")
    print(f"Depth PEFT     : {args.depth_peft_ckpt or 'N/A'}")
    print(f"Module ablation: {args.module_ablation}")
    print(f"Gaussian sigma : {args.smooth_sigma}")
    print(f"Images         : {n_images}  (abnormal={n_abnormal}, normal={n_normal})")
    print(f"Total patches  : {len(dataset)}")
    print(f"Total pixels   : {n_pixels:,}")
    if n_pixels > 0:
        gt_positive = (gt_arr == 1).sum()
        print(f"  defect px    : {gt_positive:,}  ({100 * gt_positive / n_pixels:.2f}%)")
    print()
    print(f"{'':>12} {'Pixel AUROC':>14} {'Image AUROC':>14}")
    print(f"{'':>12} {'-' * 14} {'-' * 14}")
    print(f"{'Fused':>12} {fmt(auroc_px_fused):>14} {fmt(auroc_img_fused):>14}")
    print(f"{'RGB':>12} {fmt(auroc_px_rgb):>14} {fmt(auroc_img_rgb):>14}")
    print(f"{'Depth':>12} {fmt(auroc_px_depth):>14} {fmt(auroc_img_depth):>14}")
    print()

    # ---- CSV output ----
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write(
            "ckpt,depth_peft_ckpt,module_ablation,view_id,smooth_sigma,n_images,n_abnormal,n_normal,"
            "n_patches,n_pixels,"
            "pixel_auroc_fused,pixel_auroc_rgb,pixel_auroc_depth,"
            "image_auroc_fused,image_auroc_rgb,image_auroc_depth\n"
        )
        f.write(
            f"{ckpt_path},{args.depth_peft_ckpt or ''},{args.module_ablation},"
            f"{args.view_id},{args.smooth_sigma},"
            f"{n_images},{n_abnormal},{n_normal},"
            f"{len(dataset)},{n_pixels},"
            f"{auroc_px_fused},{auroc_px_rgb},{auroc_px_depth},"
            f"{auroc_img_fused},{auroc_img_rgb},{auroc_img_depth}\n"
        )
    print(f"Results saved to: {out_csv}")


if __name__ == "__main__":
    main()
