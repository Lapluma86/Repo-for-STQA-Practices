# Cam4 Ablation Helpers

This folder contains the runnable helpers for Chapter 6 Cam4 ablations.

## 1. CF/CA post-hoc ablation

Run on an environment with PyTorch and the Cam4 test set:

```bash
python scripts/ablation/run_cam4_cf_ca.py \
  --ckpt outputs/rail_all/Cam4/20260501_014226_cam4_bs32_lr0.005_img512_ratio1.0/best_cam4.pth \
  --test_root rail_mvtec_gt_test \
  --device cuda:0
```

Outputs:

```text
outputs/rail_ablation/cam4_cf_ca/<mode>/result.json
outputs/rail_ablation/cam4_cf_ca/<mode>/scores_*.csv
outputs/rail_ablation/cam4_cf_ca/summary.csv
```

`--assist_fill zeros` is the default because Table 6.6.2 only uses RGB/Depth/Fusion
cross-conditioned AUROC. It avoids requiring the training root for isolated-branch
reference features.

## 2. Depth normalization retraining ablation

Run on the training server:

```bash
python scripts/ablation/run_cam4_depth_norm.py \
  --train_root /data1/Leaddo_data/20260327-resize512 \
  --test_root /home/root123/LF/WYM/SteelRailWay/rail_mvtec_gt_test \
  --device cuda:0
```

Use `--dry_run` first to inspect commands. Use `--eval_only --skip_existing` to
evaluate already-trained norm runs.

Outputs:

```text
outputs/rail_ablation/depth_norm/Cam4/<norm>/<timestamp>_cam4_.../best_cam4.pth
outputs/rail_ablation/depth_norm/Cam4/<norm>/eval/result.json
outputs/rail_ablation/depth_norm/Cam4/summary.csv
```

## 3. Combined thesis summary

After the above experiments finish:

```bash
python scripts/ablation/summarize_cam4_ablation.py --count_full_params
```

Output:

```text
outputs/rail_ablation/cam4_ablation_summary.csv
outputs/rail_ablation/cam4_ablation_summary.json
```

