from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset import SESSIONS

log = logging.getLogger(__name__)


class HDF5GroupedDataset(Dataset):
    """
    从HDF5文件读取的GroupedParticipantDataset

    优势：
    - 单文件顺序读取，NFS友好
    - 减少文件系统元数据操作
    - 支持按需加载或全量preload
    """

    def __init__(
        self,
        hdf5_path: str | Path,
        session_drop_prob: float = 0.0,
        preload: bool = False,
    ):
        self.hdf5_path = Path(hdf5_path)
        self.session_drop_prob = float(session_drop_prob)
        self._preload = preload
        self._cache: list[dict | None] | None = None

        # 打开HDF5文件获取元信息
        with h5py.File(self.hdf5_path, 'r') as f:
            self.n_participants = f.attrs['n_participants']
            self.split = f.attrs.get('split', 'unknown')

        log.info(f"Loaded HDF5 dataset: {self.hdf5_path}")
        log.info(f"  Participants: {self.n_participants}")
        log.info(f"  Split: {self.split}")

        # 探测特征维度
        self._feature_dims: dict[str, int] | None = None

        # 如果启用preload，立即加载所有数据到内存
        if self._preload:
            self._do_preload()

    def _do_preload(self):
        """将所有数据加载到内存"""
        from tqdm import tqdm

        log.info("Preloading HDF5 data into RAM...")
        self._cache = [None] * self.n_participants

        with h5py.File(self.hdf5_path, 'r') as f:
            for i in tqdm(range(self.n_participants), desc="Preload HDF5", dynamic_ncols=True):
                self._cache[i] = self._load_participant_from_h5(f, i)

        gb = self._estimate_cache_bytes() / 1024**3
        log.info(f"Preloaded {self.n_participants} participants ({gb:.1f} GB in RAM)")

    def _estimate_cache_bytes(self) -> int:
        """估算缓存占用的内存"""
        total = 0
        if self._cache is None:
            return 0
        for sample in self._cache:
            if sample is None:
                continue
            for sess in sample.get("sessions", []):
                if sess is None:
                    continue
                for v in sess.values():
                    if isinstance(v, torch.Tensor):
                        total += v.nelement() * v.element_size()
                    elif isinstance(v, dict):
                        for vv in v.values():
                            if isinstance(vv, torch.Tensor):
                                total += vv.nelement() * vv.element_size()
        return total

    def _load_participant_from_h5(self, f: h5py.File, idx: int) -> dict[str, Any]:
        """从HDF5文件加载单个参与者"""
        grp = f[f"p_{idx:05d}"]

        # 加载元信息和标签
        sample = {
            "anon_pid": grp.attrs['anon_pid'],
            "anon_school": grp.attrs['anon_school'],
            "anon_class": grp.attrs['anon_class'],
            "y_a1": torch.from_numpy(grp['y_a1'][:]),
            "y_a2": torch.from_numpy(grp['y_a2'][:]),
            "aux_attrs": torch.from_numpy(grp['aux_attrs'][:]),
            "session_valid": grp['session_valid'][:],
            "session_names": SESSIONS,
        }

        # 加载所有session
        sessions = []
        for sess_idx in range(4):  # 4个session: A01, B01, B02, B03
            sess_key = f"s{sess_idx}"
            if sess_key in grp:
                sess_data = self._load_session_from_h5(grp[sess_key])
                sessions.append(sess_data)
            else:
                sessions.append(None)

        sample["sessions"] = sessions
        return sample

    def _load_session_from_h5(self, sess_grp: h5py.Group) -> dict[str, Any]:
        """从HDF5加载单个session"""
        # 加载音频特征
        audio_groups = {}
        for name in sess_grp['audio'].keys():
            audio_groups[name] = torch.from_numpy(sess_grp['audio'][name][:])

        # 加载音频池化特征
        audio_pooled_groups = {}
        audio_pooled_present = {}
        if 'audio_pooled' in sess_grp:
            for name in sess_grp['audio_pooled'].keys():
                audio_pooled_groups[name] = torch.from_numpy(sess_grp['audio_pooled'][name][:])
                audio_pooled_present[name] = sess_grp['audio_pooled'].attrs.get(f'{name}_present', False)

        # 加载视频特征
        video_groups = {}
        for name in sess_grp['video'].keys():
            video_groups[name] = torch.from_numpy(sess_grp['video'][name][:])

        return {
            "audio_groups": audio_groups,
            "audio_pooled_groups": audio_pooled_groups,
            "video_groups": video_groups,
            "mask_audio": torch.from_numpy(sess_grp['mask_audio'][:]),
            "mask_video": torch.from_numpy(sess_grp['mask_video'][:]),
            "vad_signal": torch.from_numpy(sess_grp['vad_signal'][:]),
            "qc_quality": torch.from_numpy(sess_grp['qc_quality'][:]),
            "audio_pooled_present": audio_pooled_present,
            "session_idx": sess_grp.attrs['session_idx'],
            "seq_len": sess_grp.attrs['seq_len'],
            "session": sess_grp.attrs['session'],
        }

    def __len__(self) -> int:
        return self.n_participants

    def __getitem__(self, idx: int) -> dict[str, Any]:
        # 如果已preload，直接从缓存读取
        if self._cache is not None:
            sample = self._cache[idx]
        else:
            # 按需从HDF5读取
            with h5py.File(self.hdf5_path, 'r') as f:
                sample = self._load_participant_from_h5(f, idx)

        # 训练时应用session dropout
        if self.split == "train" and self.session_drop_prob > 0.0:
            return self._apply_session_dropout(sample)
        return sample

    def _apply_session_dropout(self, sample: dict[str, Any]) -> dict[str, Any]:
        """训练时随机丢弃一个session"""
        valid_indices = [
            idx for idx, is_valid in enumerate(sample["session_valid"].tolist())
            if is_valid and sample["sessions"][idx] is not None
        ]
        if len(valid_indices) <= 1 or np.random.random() >= self.session_drop_prob:
            return sample

        drop_idx = int(np.random.choice(valid_indices))
        sessions = list(sample["sessions"])
        sessions[drop_idx] = None
        session_valid = np.array(sample["session_valid"], copy=True)
        session_valid[drop_idx] = False

        return {
            **sample,
            "sessions": sessions,
            "session_valid": session_valid,
        }

    @property
    def is_preloaded(self) -> bool:
        return self._cache is not None

    @property
    def feature_dims(self) -> dict[str, int]:
        """获取特征维度信息"""
        if self._feature_dims is None:
            self._feature_dims = self._probe_dims()
        return self._feature_dims

    def _probe_dims(self) -> dict[str, int]:
        """从第一个样本探测特征维度"""
        dims: dict[str, int] = {}

        with h5py.File(self.hdf5_path, 'r') as f:
            # 找到第一个有效的session
            first_grp = f["p_00000"]
            for sess_idx in range(4):
                sess_key = f"s{sess_idx}"
                if sess_key in first_grp:
                    sess_grp = first_grp[sess_key]

                    # 探测音频特征维度
                    if 'audio' in sess_grp:
                        for name in sess_grp['audio'].keys():
                            dims[name] = sess_grp['audio'][name].shape[1]

                    # 探测音频池化特征维度
                    if 'audio_pooled' in sess_grp:
                        for name in sess_grp['audio_pooled'].keys():
                            dims[name] = sess_grp['audio_pooled'][name].shape[0]

                    # 探测视频特征维度
                    if 'video' in sess_grp:
                        for name in sess_grp['video'].keys():
                            dims[name] = sess_grp['video'][name].shape[1]

                    break  # 只需要探测一个session即可

        return dims
