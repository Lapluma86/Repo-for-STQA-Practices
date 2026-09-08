#!/usr/bin/env python3
"""
将分散的特征文件打包成HDF5，加速读取

使用：
# 打包训练集（全量）
python scripts/pack_features.py \
  --manifest /data1/AdoDas/Train/train.csv \
  --feature-root /data1/AdoDas \
  --split train \
  --output /data1/AdoDas/train_packed.h5

# 打包验证集
python scripts/pack_features.py \
  --manifest /data1/AdoDas/Val/val.csv \
  --feature-root /data1/AdoDas \
  --split val \
  --output /data1/AdoDas/val_packed.h5

# 仅打包前 100 人用于 debug
python scripts/pack_features.py \
  --manifest /data1/AdoDas/Train/train.csv \
  --feature-root /data1/AdoDas \
  --split train \
  --output /data1/AdoDas/train_debug.h5 \
  --max-participants 100
"""

import argparse
import h5py
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import sys
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from common.data.dataset import FeatureConfig
from common.data.grouped_dataset import GroupedParticipantDataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def pack_dataset(manifest_path, feature_root, split, output_path, compression='gzip',
                 config_dict=None, max_participants=0):
    """将dataset打包成HDF5"""

    # 创建配置
    if config_dict is not None:
        feature_selection = config_dict.get('feature_selection', {})
        cfg = FeatureConfig(**feature_selection, feature_root=feature_root)
    else:
        cfg = FeatureConfig(feature_root=feature_root)

    # 加载dataset
    log.info(f"Loading dataset from {manifest_path}")
    ds = GroupedParticipantDataset(manifest_path, cfg, split, session_drop_prob=0.0,
                                   max_participants=max_participants)

    log.info(f"Packing {len(ds)} participants to {output_path}")
    log.info(f"Compression: {compression}")

    with h5py.File(output_path, 'w') as f:
        # 保存元信息
        f.attrs['n_participants'] = len(ds)
        f.attrs['split'] = split

        errors = 0
        for i in tqdm(range(len(ds)), desc="Packing", dynamic_ncols=True):
            try:
                sample = ds._load_participant(i)

                # 为每个participant创建一个group
                grp = f.create_group(f"p_{i:05d}")

                # 保存元信息
                grp.attrs['anon_pid'] = sample['anon_pid']
                grp.attrs['anon_school'] = sample['anon_school']
                grp.attrs['anon_class'] = sample['anon_class']

                # 保存标签
                grp.create_dataset('y_a1', data=sample['y_a1'].numpy())
                grp.create_dataset('y_a2', data=sample['y_a2'].numpy())
                grp.create_dataset('aux_attrs', data=sample['aux_attrs'].numpy())
                grp.create_dataset('session_valid', data=sample['session_valid'])

                # 保存每个session
                for sess_idx, sess_data in enumerate(sample['sessions']):
                    if sess_data is None:
                        continue

                    sess_grp = grp.create_group(f"s{sess_idx}")

                    # 保存音频特征
                    audio_grp = sess_grp.create_group('audio')
                    for name, tensor in sess_data['audio_groups'].items():
                        audio_grp.create_dataset(
                            name,
                            data=tensor.numpy(),
                            compression=compression,
                            compression_opts=4 if compression == 'gzip' else None
                        )

                    # 保存音频池化特征
                    if sess_data['audio_pooled_groups']:
                        pooled_grp = sess_grp.create_group('audio_pooled')
                        for name, tensor in sess_data['audio_pooled_groups'].items():
                            pooled_grp.create_dataset(name, data=tensor.numpy())
                            pooled_grp.attrs[f'{name}_present'] = sess_data['audio_pooled_present'].get(name, False)

                    # 保存视频特征
                    video_grp = sess_grp.create_group('video')
                    for name, tensor in sess_data['video_groups'].items():
                        video_grp.create_dataset(
                            name,
                            data=tensor.numpy(),
                            compression=compression,
                            compression_opts=4 if compression == 'gzip' else None
                        )

                    # 保存掩码和辅助信号
                    sess_grp.create_dataset('mask_audio', data=sess_data['mask_audio'].numpy())
                    sess_grp.create_dataset('mask_video', data=sess_data['mask_video'].numpy())
                    sess_grp.create_dataset('vad_signal', data=sess_data['vad_signal'].numpy())
                    sess_grp.create_dataset('qc_quality', data=sess_data['qc_quality'].numpy())

                    # 保存元信息
                    sess_grp.attrs['session_idx'] = sess_data['session_idx']
                    sess_grp.attrs['seq_len'] = sess_data['seq_len']
                    sess_grp.attrs['session'] = sess_data['session']

            except Exception as e:
                errors += 1
                log.warning(f"Failed to pack participant {i}: {e}")
                if errors <= 3:
                    import traceback
                    traceback.print_exc()

        if errors > 0:
            log.warning(f"Failed to pack {errors}/{len(ds)} participants")

    file_size_gb = Path(output_path).stat().st_size / 1024**3
    log.info(f"Done! File size: {file_size_gb:.2f} GB")
    log.info(f"Compression ratio: ~{file_size_gb / len(ds) * 1000:.1f} MB/participant")


def main():
    parser = argparse.ArgumentParser(
        description="Pack scattered feature files into HDF5 for faster loading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Pack training set
  python scripts/pack_features.py \\
    --manifest /data1/AdoDas/Train/train.csv \\
    --feature-root /data1/AdoDas \\
    --split train \\
    --output /data1/AdoDas/train_packed.h5

  # Pack validation set
  python scripts/pack_features.py \\
    --manifest /data1/AdoDas/Val/val.csv \\
    --feature-root /data1/AdoDas \\
    --split val \\
    --output /data1/AdoDas/val_packed.h5

  # Debug subset (100 participants only)
  python scripts/pack_features.py \\
    --manifest /data1/AdoDas/Train/train.csv \\
    --feature-root /data1/AdoDas \\
    --split train \\
    --output /data1/AdoDas/train_debug.h5 \\
    --max-participants 100
        """
    )
    parser.add_argument('--manifest', required=True, help='Path to manifest CSV')
    parser.add_argument('--feature-root', required=True, help='Path to feature directory')
    parser.add_argument('--split', required=True, help='Split name (train/val/test)')
    parser.add_argument('--output', required=True, help='Output HDF5 file path')
    parser.add_argument('--compression', default='gzip', choices=['gzip', 'lzf', None],
                        help='Compression algorithm (default: gzip)')
    parser.add_argument('--config', help='Optional: path to config YAML to override feature selection')
    parser.add_argument('--max-participants', type=int, default=0,
                        help='Only pack first N participants (0=all)')

    args = parser.parse_args()

    config_dict = None
    if args.config:
        import yaml
        with open(args.config) as f:
            config_dict = yaml.safe_load(f)

    pack_dataset(
        manifest_path=args.manifest,
        feature_root=args.feature_root,
        split=args.split,
        output_path=args.output,
        compression=args.compression,
        config_dict=config_dict,
        max_participants=args.max_participants,
    )


if __name__ == '__main__':
    main()
