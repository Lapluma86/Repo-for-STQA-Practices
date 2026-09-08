from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from multiprocessing import Pool
from functools import partial

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .feature_io import SequenceData, load_egemaps_pooled, load_sequence

log = logging.getLogger(__name__)


SESSIONS = ["A01", "B01", "B02", "B03"]
SESSION_TO_IDX = {s: i for i, s in enumerate(SESSIONS)}
ITEM_COLS = [f"d{i:02d}" for i in range(1, 22)]
A1_COLS = ["y_D", "y_A", "y_S"]
AUX_ATTR_COLS = ["Family structure", "Only child status", "Parental favoritism",
                 "Academic performance change", "Emotional state change"]
POOLED_AUDIO_FEATURES = {"egemaps"}

# 逻辑 split 名 → feature_root 下的相对子路径
SPLIT_DATA_PATH = {
    "train": "Train/train/train",
    "val": "Val/val/val",
    "test_hidden": "Test/test/test_hidden",
}

'''
此文件实现的核心功能：
1. 多模态对齐
2. 确实处理
3. 预加载机制
'''

@dataclass
class FeatureConfig:
    feature_root: str = "/data1/AdoDas"
    audio_features: list[str] = field(
        default_factory=lambda: ["mel_mfcc", "vad", "egemaps", "ssl_embed"]
    )
    video_features: list[str] = field(
        default_factory=lambda: [
            "headpose_geom", "face_behavior", "qc_stats", "vad_agg",
            "body_pose", "global_motion", "vision_ssl_embed",
        ]
    )
    # 指定使用哪个预训练模型的特征，如果audio_features或video_features中包含"ssl_embed"或"vision_ssl_embed"，则使用对应的tag加载特征文件
    audio_ssl_model_tag: str = "chinese-hubert-base" 
    video_ssl_model_tag: str = "dinov2-base"
    # 对齐时间步长和容忍度
    grid_step_ms: float = 40.0
    tolerance_ms: float = 25.0


    mask_policy: str = "and_core" # 必须要核心特征都存在，才算该时间步有效，可选and_core，or，require_k
    core_audio: list[str] = field(default_factory=lambda: ["mel_mfcc", "vad"])
    core_video: list[str] = field(default_factory=lambda: ["face_behavior", "headpose_geom"])

    @property
    def audio_sequence_features(self) -> list[str]:
        return [name for name in self.audio_features if name not in POOLED_AUDIO_FEATURES]

    @property
    def audio_pooled_features(self) -> list[str]:
        return [name for name in self.audio_features if name in POOLED_AUDIO_FEATURES]


# 为同一时间网格的每一个点，找到最近的特征时间戳，并返回对应的索引和距离
def _nearest_indices(grid: np.ndarray, timestamps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    idx = np.searchsorted(timestamps, grid, side="left")
    idx = np.clip(idx, 0, len(timestamps) - 1)

    idx_left = np.clip(idx - 1, 0, len(timestamps) - 1)
    dist_right = np.abs(grid - timestamps[idx])
    dist_left = np.abs(grid - timestamps[idx_left])
    use_left = dist_left < dist_right
    best_idx = np.where(use_left, idx_left, idx)
    best_dist = np.where(use_left, dist_left, dist_right)
    return best_idx, best_dist


def align_to_grid(
    groups: dict[str, SequenceData],
    grid_step_ms: float = 40.0,
    tolerance_ms: float = 25.0,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, int]:
    """
    将多个不同采样率的特征序列对齐到统一时间网格
    
    核心思想：
    1. 找到所有特征的时间范围[t_min, t_max]
    2. 在此范围内生成等间隔网格（步长40ms）
    3. 每个特征通过最近邻采样对齐到网格
    4. 超出容差的点标记为无效
    """    
    if not groups:
        raise ValueError("No feature groups supplied for alignment")

    t_min = min(seq.timestamps_ms[0] for seq in groups.values())
    t_max = max(seq.timestamps_ms[-1] for seq in groups.values())
    grid = np.arange(t_min, t_max + grid_step_ms * 0.5, grid_step_ms) # 生成时间网格，末尾加半步长确保覆盖最后一个点
    T = len(grid)

    aligned_feats: dict[str, np.ndarray] = {}
    aligned_masks: dict[str, np.ndarray] = {}

    for name, seq in groups.items():
        best_idx, best_dist = _nearest_indices(grid, seq.timestamps_ms) # 找到每个网格点最近的特征索引和距离
        within = best_dist <= tolerance_ms # 判断是否在容差范围内
        aligned_feats[name] = seq.features[best_idx]  # (T, D)
        aligned_masks[name] = seq.valid_mask[best_idx] & within # 只有原始有效且在容差范围内的点才算对齐后有效

    return aligned_feats, aligned_masks, grid, T


class MultimodalDataset(Dataset):
    
    def __init__(
        self,
        manifest_path: str | Path,
        cfg: FeatureConfig,
        split: str,
    ) -> None:
        self.cfg = cfg
        self.split = split
        self.root = Path(cfg.feature_root) / SPLIT_DATA_PATH.get(split, split)

        # 加载manifest，检查必要的列
        self.manifest = pd.read_csv(manifest_path)
        required = {"anon_school", "anon_class", "anon_pid", "session"}
        missing = required - set(self.manifest.columns)
        if missing:
            raise KeyError(f"Manifest missing columns: {missing}")

        self._feature_dims: dict[str, int] | None = None

        self._cache: list[dict[str, Any] | None] | None = None

    @property
    def feature_dims(self) -> dict[str, int]:
        """Lazy-compute feature dims from the first sample."""
        if self._feature_dims is None:
            self._feature_dims = self._probe_dims()
        return self._feature_dims

    def _probe_dims(self) -> dict[str, int]:
        row = self.manifest.iloc[0] # 从第一个样本获取特征维度信息，因为不同SSL模型可能维度不同
        dims: dict[str, int] = {}
        # 通过探测实际文件来获取维度，而不是依赖固定的配置，这样可以适应不同预训练模型的特征维度
        for name, seq in self._load_raw_groups(row, "audio").items():
            dims[name] = seq.features.shape[1]
        for name, seq in self._load_raw_groups(row, "video").items():
            dims[name] = seq.features.shape[1]
        # 对于池化特征，直接加载并获取维度
        if "egemaps" in self.cfg.audio_pooled_features:
            eg = load_egemaps_pooled(
                self.root, "",
                str(row["anon_school"]), str(row["anon_class"]),
                str(row["anon_pid"]), str(row["session"]),
            )
            if eg is not None:
                dims["egemaps"] = len(eg)
        return dims

    @staticmethod
    def _compute_modality_mask(
        mask_parts: list[np.ndarray], # 每个特征的有效性掩码列表
        mask_names: list[str], # 每个掩码对应的特征名称列表
        core_names: list[str], # 核心特征名称列表
        policy: str, # 掩码融合策略：or，and_core，require_k
        T: int, # 序列长度
    ) -> np.ndarray:
        if not mask_parts:
            return np.zeros(T, dtype=bool) # 如果没有任何特征，整个模态都无效

        if policy == "or":
            return np.any(np.stack(mask_parts), axis=0) # 只要任一特征有效，该时间步就有效

        if policy == "and_core":
            core_masks = [
                m for m, n in zip(mask_parts, mask_names) if n in core_names
            ]
            if core_masks:
                return np.all(np.stack(core_masks), axis=0) # 核心特征必须全部有效
            return np.any(np.stack(mask_parts), axis=0)

        if policy == "require_k":
            core_masks = [
                m for m, n in zip(mask_parts, mask_names) if n in core_names
            ]
            k = max(1, len(core_names))
            if core_masks:
                return np.sum(np.stack(core_masks), axis=0) >= k
            return np.any(np.stack(mask_parts), axis=0)

        raise ValueError(f"Unknown mask_policy: {policy!r}")


    # 预加载机制
    def preload(self, desc: str | None = None, num_workers: int = 8) -> float:
        n = len(self)
        if desc is None:
            desc = f"Preload {self.split}"
        self._cache = [None] * n
        errors = 0

        # 多进程并行加载
        if num_workers > 1:
            with Pool(num_workers) as pool:
                results = list(tqdm(
                    pool.imap(self._load_sample_safe, range(n)),
                    total=n,
                    desc=desc,
                    dynamic_ncols=True
                ))

            for i, result in enumerate(results):
                if result is not None:
                    self._cache[i] = result
                else:
                    errors += 1
        else:
            # 单进程加载（原逻辑）
            for i in tqdm(range(n), desc=desc, dynamic_ncols=True):
                try:
                    self._cache[i] = self._load_sample(i)
                except Exception as exc:
                    errors += 1
                    if errors <= 3:
                        log.warning(f"Preload: sample {i} failed: {exc}")

        if errors > 0:
            log.warning(f"Preload: {errors}/{n} samples failed and will be skipped")
        gb = self._estimate_cache_bytes() / 1024**3
        log.info(f"Preloaded {n - errors}/{n} samples ({gb:.1f} GB in RAM)")
        return gb

    def _load_sample_safe(self, idx: int) -> dict[str, Any] | None:
        """多进程安全的样本加载（捕获异常）"""
        try:
            return self._load_sample(idx)
        except Exception as exc:
            log.warning(f"Preload: sample {idx} failed: {exc}")
            return None

    def _estimate_cache_bytes(self) -> int:
        total = 0
        if self._cache is None:
            return 0
        for sample in self._cache:
            if sample is None:
                continue
            for v in sample.values():
                if isinstance(v, torch.Tensor):
                    total += v.nelement() * v.element_size()
                elif isinstance(v, dict):
                    for vv in v.values():
                        if isinstance(vv, torch.Tensor):
                            total += vv.nelement() * vv.element_size()
        return total

    @property
    def is_preloaded(self) -> bool:
        return self._cache is not None

    def _load_raw_groups(            
        self, row: pd.Series, modality: str
    ) -> dict[str, SequenceData]:
        cfg = self.cfg
        feat_list = cfg.audio_sequence_features if modality == "audio" else cfg.video_features
        groups: dict[str, SequenceData] = {}
        for feat_name in feat_list:
            tag: str | None = None
            if feat_name == "ssl_embed":
                tag = cfg.audio_ssl_model_tag
            elif feat_name == "vision_ssl_embed":
                tag = cfg.video_ssl_model_tag
            try:
                seq = load_sequence(
                    self.root, "",
                    str(row["anon_school"]), str(row["anon_class"]),
                    str(row["anon_pid"]),
                    modality, feat_name, str(row["session"]),
                    model_tag=tag,
                )
                groups[feat_name] = seq
            except FileNotFoundError:
                pass 
        return groups

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self._cache is not None and self._cache[idx] is not None:
            return self._cache[idx]
        return self._load_sample(idx)

    def _load_sample(self, idx: int) -> dict[str, Any]:
        """
        加载单个样本的完整流程：
        1. 加载原始特征
        2. 时间对齐
        3. 计算掩码
        4. 提取辅助信号（VAD、质量分数）
        5. 加载标签
        6. 零填充缺失特征
        """
        row = self.manifest.iloc[idx]
        cfg = self.cfg

        # 加载原始特征序列，得到一个字典，键是"audio/feat_name"或"video/feat_name"，值是SequenceData对象
        audio_raw = self._load_raw_groups(row, "audio")
        video_raw = self._load_raw_groups(row, "video")

        all_groups = {}
        for k, v in audio_raw.items():
            all_groups[f"audio/{k}"] = v
        for k, v in video_raw.items():
            all_groups[f"video/{k}"] = v # 合并音频和视频特征到一个字典，键带上前缀区分模态

        if not all_groups:
            raise RuntimeError(
                f"No features loaded for {row['anon_pid']} session {row['session']}"
            )

        # 对齐到统一时间网格，得到对齐后的特征和掩码，以及网格时间戳和长度T
        aligned_feats, aligned_masks, grid_ms, T = align_to_grid(
            all_groups, cfg.grid_step_ms, cfg.tolerance_ms
        )

        # 分离音频、视频特征，收集掩码
        audio_groups: dict[str, torch.Tensor] = {}
        video_groups: dict[str, torch.Tensor] = {}
        audio_mask_parts: list[np.ndarray] = []
        audio_mask_names: list[str] = []
        video_mask_parts: list[np.ndarray] = []
        video_mask_names: list[str] = []

        for key, feat in aligned_feats.items():
            modality, name = key.split("/", 1)
            mask = aligned_masks[key]
            t = torch.from_numpy(feat.astype(np.float32))
            if modality == "audio":
                audio_groups[name] = t
                audio_mask_parts.append(mask)
                audio_mask_names.append(name)
            else:
                video_groups[name] = t
                video_mask_parts.append(mask)
                video_mask_names.append(name)

        # 计算模态掩码，根据配置的策略和核心特征要求，得到每个时间步该模态是否有效
        mask_audio = self._compute_modality_mask(
            audio_mask_parts, audio_mask_names, cfg.core_audio, cfg.mask_policy, T
        )
        mask_video = self._compute_modality_mask(
            video_mask_parts, video_mask_names, cfg.core_video, cfg.mask_policy, T
        )

        # 从对齐后的特征中提取VAD信号和质量分数，如果存在的话，乘以对应的掩码确保无效时间步的信号为0
        vad_signal = np.zeros(T, dtype=np.float32)
        if "audio/vad" in aligned_feats:
            v = aligned_feats["audio/vad"]
            # vad第一列是概率值，乘以掩码后得到最终的VAD信号，过滤无效帧
            vad_signal = v[:, 0].astype(np.float32) * aligned_masks["audio/vad"].astype(np.float32)
        elif "video/vad_agg" in aligned_feats:
            # 若音频VAD缺失，则使用视频VAD的聚合版本作为替代，虽然可能不如音频VAD准确，但至少提供一些活动信息
            v = aligned_feats["video/vad_agg"]
            vad_signal = v[:, 0].astype(np.float32) * aligned_masks["video/vad_agg"].astype(np.float32)

        # 从视频质量控制统计中提取质量分数，如果存在的话，乘以掩码确保无效时间步的质量分数为0
        qc_quality = np.zeros(T, dtype=np.float32)
        if "video/qc_stats" in aligned_feats:
            v = aligned_feats["video/qc_stats"]
            qc_quality = v[:, 0].astype(np.float32) * aligned_masks["video/qc_stats"].astype(np.float32)

        # 加载池化特征，如egemaps，如果存在的话，放入audio_pooled_groups字典，并记录其存在性到pooled_presence字典中，以便后续使用时知道哪些池化特征是可用的
        audio_pooled_groups: dict[str, torch.Tensor] = {}
        pooled_presence: dict[str, bool] = {}
        if "egemaps" in cfg.audio_pooled_features:
            egemaps = load_egemaps_pooled(
                self.root, "",
                str(row["anon_school"]), str(row["anon_class"]),
                str(row["anon_pid"]), str(row["session"]),
            )
            dims = self.feature_dims
            # 如果缺失则用全零向量填充，并且标记为不存在，这样模型在使用时可以区分是缺失还是实际的零值
            audio_pooled_groups["egemaps"] = (
                torch.from_numpy(egemaps)
                if egemaps is not None
                else torch.zeros(dims.get("egemaps", 88))
            )
            pooled_presence["egemaps"] = egemaps is not None

        session_idx = SESSION_TO_IDX.get(str(row["session"]), 0) # 会话索引，用来区分不同的会话类型，默认0

        # 加载两个任务的集合
        y_a1 = np.array(
            [float(row.get(c, -1)) for c in A1_COLS], dtype=np.float32
        )
        y_a2 = np.array(
            [float(row.get(c, -1)) for c in ITEM_COLS], dtype=np.float32
        )

        # 零填充缺失特征，保证所有样本特征集一致，维度也一致，这样模型输入时就不需要特殊处理缺失特征的情况
        dims = self.feature_dims
        for name in cfg.audio_features:
            if name not in audio_groups and name not in cfg.audio_pooled_features and name in dims:
                audio_groups[name] = torch.zeros(T, dims[name])
        for name in cfg.video_features:
            if name not in video_groups and name in dims:
                video_groups[name] = torch.zeros(T, dims[name])

        return {
            "audio_groups": audio_groups, # 音频序列特征字典
            "audio_pooled_groups": audio_pooled_groups, # 音频池化特征字典
            "video_groups": video_groups,   # 视频序列特征字典
            "mask_audio": torch.from_numpy(mask_audio), # 音频有效帧掩码
            "mask_video": torch.from_numpy(mask_video), # 视频有效帧掩码
            "vad_signal": torch.from_numpy(vad_signal), # VAD信号，表示每个时间步是否有语音活动，值为概率
            "qc_quality": torch.from_numpy(qc_quality), # 视频质量分数
            "audio_pooled_present": pooled_presence, # 音频池化特征存在性标记
            "session_idx": session_idx, # 会话索引
            "y_a1": torch.from_numpy(y_a1), # 任务1标签
            "y_a2": torch.from_numpy(y_a2), # 任务2标签
            "seq_len": T, # 序列长度
            "anon_pid": str(row["anon_pid"]), # 匿名用户ID
            "session": str(row["session"]), # 会话ID
        }

# 对于变长序列的批处理，进行零填充并生成掩码，确保模型输入的一致性，同时保留原始序列长度信息以供后续使用
def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    B = len(batch)
    T_max = max(b["seq_len"] for b in batch)


    audio_names = list(batch[0]["audio_groups"].keys())
    pooled_audio_names = list(batch[0]["audio_pooled_groups"].keys())
    video_names = list(batch[0]["video_groups"].keys())


    def _pad_groups(names: list[str], key: str) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}
        for n in names:
            D = batch[0][key][n].shape[-1] # 获取特征维度，假设同一特征在不同样本中维度一致
            t = torch.zeros(B, T_max, D) # 初始化为0
            for i, b in enumerate(batch):
                L = b["seq_len"]
                t[i, :L] = b[key][n] # 填充有效部分，超出部分保持为0
            result[n] = t
        return result

# 对于一维信号（如掩码、VAD、质量分数），直接创建二维张量并填充，保持与序列特征的时间维度一致，方便模型输入时对齐
    def _pad_1d(key: str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        t = torch.zeros(B, T_max, dtype=dtype)
        for i, b in enumerate(batch):
            L = b["seq_len"]
            t[i, :L] = b[key]
        return t

    pad_mask = torch.ones(B, T_max, dtype=torch.bool)
    for i, b in enumerate(batch):
        pad_mask[i, : b["seq_len"]] = False

    return {
        "audio_groups": _pad_groups(audio_names, "audio_groups"),
        "audio_pooled_groups": {
            name: torch.stack([b["audio_pooled_groups"][name] for b in batch])
            for name in pooled_audio_names
        },
        "video_groups": _pad_groups(video_names, "video_groups"),
        "mask_audio": _pad_1d("mask_audio", torch.bool),
        "mask_video": _pad_1d("mask_video", torch.bool),
        "pad_mask": pad_mask,
        "vad_signal": _pad_1d("vad_signal"),
        "qc_quality": _pad_1d("qc_quality"),
        "audio_pooled_present": {
            name: torch.tensor(
                [b["audio_pooled_present"].get(name, False) for b in batch],
                dtype=torch.bool,
            )
            for name in pooled_audio_names
        },
        "session_idx": torch.tensor([b["session_idx"] for b in batch], dtype=torch.long),
        "y_a1": torch.stack([b["y_a1"] for b in batch]),
        "y_a2": torch.stack([b["y_a2"] for b in batch]),
        "seq_len": torch.tensor([b["seq_len"] for b in batch], dtype=torch.long),
        "anon_pid": [b["anon_pid"] for b in batch],
        "session": [b["session"] for b in batch],
    }
