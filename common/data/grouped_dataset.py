from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .dataset import (
    SESSIONS, SESSION_TO_IDX, ITEM_COLS, A1_COLS, AUX_ATTR_COLS,
    FeatureConfig, SPLIT_DATA_PATH, align_to_grid,
)
from .feature_io import SequenceData, load_egemaps_pooled, load_sequence

log = logging.getLogger(__name__)


class GroupedParticipantDataset(Dataset):
    """
    按参与者分组的数据集，每个样本包含一个人的所有会话（最多4个）

    核心：
    1. 纵向建模：捕捉同一个体在不同时间点的心理状态变化
    2. 会话dropout：训练时随机丢弃会话，防止过拟合特定测试时间点
    3. 标签共享：同一参与者的所有会话共享相同的DASS-21标签
    """
    def __init__(
        self,
        manifest_path: str | Path,
        cfg: FeatureConfig,
        split: str,
        session_drop_prob: float = 0.0,
        max_participants: int = 0,
    ) -> None:
        self.cfg = cfg
        self.split = split
        self.session_drop_prob = float(session_drop_prob)
        self.root = Path(cfg.feature_root) / SPLIT_DATA_PATH.get(split, split)

        manifest = pd.read_csv(manifest_path)

        # 按(学校, 班级, 学生ID)分组，将同一个人的多次测试聚合在一起
        group_cols = ["anon_school", "anon_class", "anon_pid"]
        grouped = manifest.groupby(group_cols)

        self.participants: list[dict[str, Any]] = []
        for (school, cls, pid), group in grouped:
            # 构建会话字典：{"A01": row1, "B01": row2, ...}
            sess_rows = {}
            for _, row in group.iterrows():
                sess = str(row["session"])
                sess_rows[sess] = row

            # 提取标签（同一参与者的所有会话标签相同，取任意一行即可）
            any_row = group.iloc[0]
            y_a1 = np.array([float(any_row.get(c, -1)) for c in A1_COLS], dtype=np.float32)  # 维度级标签：[y_D, y_A, y_S]
            y_a2 = np.array([float(any_row.get(c, -1)) for c in ITEM_COLS], dtype=np.float32)  # 题目级标签：[d01-d21]

            # 提取辅助属性（同一参与者的所有会话共享相同的辅助属性）
            aux_attrs = np.array([float(any_row.get(c, -1)) for c in AUX_ATTR_COLS], dtype=np.float32)

            self.participants.append({
                "anon_school": str(school),
                "anon_class": str(cls),
                "anon_pid": str(pid),
                "sess_rows": sess_rows,
                "y_a1": y_a1,
                "y_a2": y_a2,
                "aux_attrs": aux_attrs,
            })

        if max_participants > 0 and len(self.participants) > max_participants:
            self.participants = self.participants[:max_participants]
            log.info(f"Truncated dataset to {max_participants} participants (debug mode)")

        self._feature_dims: dict[str, int] | None = None
        self._cache: list[dict | None] | None = None

    @property
    def feature_dims(self) -> dict[str, int]:
        if self._feature_dims is None:
            self._feature_dims = self._probe_dims()
        return self._feature_dims

    def _probe_dims(self) -> dict[str, int]:
        info = self.participants[0]
        sess_rows = info["sess_rows"]
        any_sess = list(sess_rows.keys())[0]
        row = sess_rows[any_sess]
        dims: dict[str, int] = {}
        for name, seq in self._load_raw_groups(row, "audio").items():
            dims[name] = seq.features.shape[1]
        for name, seq in self._load_raw_groups(row, "video").items():
            dims[name] = seq.features.shape[1]
        if "egemaps" in self.cfg.audio_pooled_features:
            eg = load_egemaps_pooled(
                self.root, "",
                str(row["anon_school"]), str(row["anon_class"]),
                str(row["anon_pid"]), str(row["session"]),
            )
            if eg is not None:
                dims["egemaps"] = len(eg)
        return dims

    # 加载原始特征组，返回一个字典，键是特征名，值是SequenceData对象
    def _load_raw_groups(self, row, modality: str) -> dict[str, SequenceData]:
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

    # 计算模态掩码的核心函数，根据多个特征的掩码和策略合成一个模态级掩码
    def _compute_modality_mask(
        self, mask_parts, mask_names, core_names, policy, T
    ) -> np.ndarray:
        if not mask_parts:
            return np.zeros(T, dtype=bool)
        if policy == "or":
            return np.any(np.stack(mask_parts), axis=0)
        if policy == "and_core":
            core_masks = [m for m, n in zip(mask_parts, mask_names) if n in core_names]
            if core_masks:
                return np.all(np.stack(core_masks), axis=0)
            return np.any(np.stack(mask_parts), axis=0)
        if policy == "require_k":
            k = max(1, len(core_names))
            stacked = np.stack(mask_parts)
            return np.sum(stacked, axis=0) >= k
        raise ValueError(f"Unknown mask_policy: {policy!r}")

    def _load_single_session(self, row) -> dict[str, Any] | None:
        """
        加载单个会话的特征数据

        返回：
            成功时返回包含特征和掩码的字典，失败时返回None

        设计要点：
        - 与dataset.py的_load_sample逻辑相同，但不包含标签（标签在参与者级别）
        - 返回None而非抛出异常，允许部分会话缺失（如某次测试设备故障）
        """
        cfg = self.cfg
        try:
            # 步骤1：加载原始特征
            audio_raw = self._load_raw_groups(row, "audio")
            video_raw = self._load_raw_groups(row, "video")

            # 合并为统一字典，键名加上模态前缀
            all_groups = {}
            for k, v in audio_raw.items():
                all_groups[f"audio/{k}"] = v
            for k, v in video_raw.items():
                all_groups[f"video/{k}"] = v

            if not all_groups:
                return None  # 无任何特征时返回None，而非抛出异常

            # 步骤2：时间对齐到统一网格
            aligned_feats, aligned_masks, grid_ms, T = align_to_grid(
                all_groups, cfg.grid_step_ms, cfg.tolerance_ms
            )

            audio_groups: dict[str, torch.Tensor] = {}
            video_groups: dict[str, torch.Tensor] = {}
            audio_mask_parts, audio_mask_names = [], []
            video_mask_parts, video_mask_names = [], []

            # 步骤3：分离音频和视频特征，收集掩码
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

            # 步骤4：根据策略计算模态级掩码
            mask_audio = self._compute_modality_mask(
                audio_mask_parts, audio_mask_names, cfg.core_audio, cfg.mask_policy, T
            )
            mask_video = self._compute_modality_mask(
                video_mask_parts, video_mask_names, cfg.core_video, cfg.mask_policy, T
            )

            # 步骤5：提取VAD信号（语音活动检测）
            vad_signal = np.zeros(T, dtype=np.float32)
            if "audio/vad" in aligned_feats:
                v = aligned_feats["audio/vad"]
                vad_signal = v[:, 0].astype(np.float32) * aligned_masks["audio/vad"].astype(np.float32)
            elif "video/vad_agg" in aligned_feats:
                v = aligned_feats["video/vad_agg"]
                vad_signal = v[:, 0].astype(np.float32) * aligned_masks["video/vad_agg"].astype(np.float32)

            # 步骤6：提取质量分数
            qc_quality = np.zeros(T, dtype=np.float32)
            if "video/qc_stats" in aligned_feats:
                v = aligned_feats["video/qc_stats"]
                qc_quality = v[:, 0].astype(np.float32) * aligned_masks["video/qc_stats"].astype(np.float32)

            # 步骤7：加载池化特征（egemaps）
            dims = self.feature_dims
            audio_pooled_groups: dict[str, torch.Tensor] = {}
            pooled_presence: dict[str, bool] = {}
            if "egemaps" in cfg.audio_pooled_features:
                egemaps = load_egemaps_pooled(
                    self.root, "",
                    str(row["anon_school"]), str(row["anon_class"]),
                    str(row["anon_pid"]), str(row["session"]),
                )
                audio_pooled_groups["egemaps"] = (
                    torch.from_numpy(egemaps) if egemaps is not None
                    else torch.zeros(dims.get("egemaps", 88))
                )
                pooled_presence["egemaps"] = egemaps is not None

            # 步骤8：零填充缺失特征（确保所有会话特征集一致）
            for name in cfg.audio_features:
                if name not in audio_groups and name not in cfg.audio_pooled_features and name in dims:
                    audio_groups[name] = torch.zeros(T, dims[name])
            for name in cfg.video_features:
                if name not in video_groups and name in dims:
                    video_groups[name] = torch.zeros(T, dims[name])

            session_idx = SESSION_TO_IDX.get(str(row["session"]), 0)

            return {
                "audio_groups": audio_groups,
                "audio_pooled_groups": audio_pooled_groups,
                "video_groups": video_groups,
                "mask_audio": torch.from_numpy(mask_audio),
                "mask_video": torch.from_numpy(mask_video),
                "vad_signal": torch.from_numpy(vad_signal),
                "qc_quality": torch.from_numpy(qc_quality),
                "audio_pooled_present": pooled_presence,
                "session_idx": session_idx,
                "seq_len": T,
                "session": str(row["session"]),
            }
        except Exception as e:
            log.warning(f"Failed to load session {row.get('session', '?')} for {row.get('anon_pid', '?')}: {e}")
            return None  # 静默失败，允许部分会话缺失

    def __len__(self) -> int:
        return len(self.participants)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        获取单个参与者的数据

        返回字典包含：
        - sessions: 长度为4的列表，每个元素是一个会话的特征字典（或None）
        - session_valid: 布尔数组，标记哪些会话有效
        - y_a1, y_a2: 该参与者的标签（所有会话共享）
        """
        if self._cache is not None and self._cache[idx] is not None:
            sample = self._cache[idx]
        else:
            sample = self._load_participant(idx)

        # 训练时应用会话dropout进行数据增强
        if self.split == "train" and self.session_drop_prob > 0.0:
            return self._apply_session_dropout(sample)
        return sample

    def _load_emotion_dims(self, y_a1: np.ndarray) -> np.ndarray:
        """
        从DASS-21分数推导情绪维度（valence/arousal）

        参数:
            y_a1: [depression, anxiety, stress] 范围 [0, 3]

        返回:
            (2,) [valence, arousal] 范围 [0, 1]
            - valence（愉悦度）：抑郁分数越高，valence越低
            - arousal（激活度）：焦虑分数越高，arousal越高
        """
        depression = float(y_a1[0])
        anxiety = float(y_a1[1])

        # 归一化到 [0, 1]，处理缺失值
        if depression < 0:
            valence = 0.5  # 缺失值用中性值
        else:
            valence = 1.0 - (depression / 3.0)

        if anxiety < 0:
            arousal = 0.5
        else:
            arousal = anxiety / 3.0

        return np.array([valence, arousal], dtype=np.float32)

    def _load_participant(self, idx: int) -> dict[str, Any]:
        """
        加载一个参与者的所有会话

        关键设计：
        - 按固定顺序[A01, B01, B02, B03]组织会话，保持批次内结构一致
        - 缺失的会话用None占位，通过session_valid标记
        """
        info = self.participants[idx]
        sessions_data = []
        session_valid = []

        # 按固定顺序遍历所有可能的会话
        for sess_name in SESSIONS:  # ["A01", "B01", "B02", "B03"]
            if sess_name in info["sess_rows"]:
                data = self._load_single_session(info["sess_rows"][sess_name])
                if data is not None:
                    sessions_data.append(data)
                    session_valid.append(True)
                else:
                    sessions_data.append(None)
                    session_valid.append(False)
            else:
                sessions_data.append(None)
                session_valid.append(False)

        sample = {
            "sessions": sessions_data,  # 长度为4的列表，元素为会话字典或None
            "session_valid": np.array(session_valid, dtype=bool),  # [True, True, False, True]表示第3个会话缺失
            "y_a1": torch.from_numpy(info["y_a1"]),
            "y_a2": torch.from_numpy(info["y_a2"]),
            "aux_attrs": torch.from_numpy(info["aux_attrs"]),
            "anon_pid": info["anon_pid"],
            "anon_school": info["anon_school"],
            "anon_class": info["anon_class"],
            "session_names": SESSIONS,
        }

        # 辅助任务标签 — 仅保留 emotion_dims 作为弱正则化项 (weight ≤ 0.05)
        # emotion_cls 和 AU 已删除：前者与主任务标签同源 (deterministic 推导)，后者为占位零实现
        if self.split == "train":
            sample["auxiliary_targets"] = {
                "emotion_dims": torch.from_numpy(self._load_emotion_dims(info["y_a1"])),
            }

        return sample

    def _apply_session_dropout(self, sample: dict[str, Any]) -> dict[str, Any]:
        """
        训练时随机丢弃一个会话，用于数据增强

        动机：
        - 防止模型过拟合特定测试时间点（如总是依赖B03的数据）
        - 提高模型对缺失会话的鲁棒性（测试时某些会话可能不可用）

        策略：
        - 只在有2个及以上有效会话时才dropout（保证至少有1个会话）
        - 以session_drop_prob概率触发
        """
        valid_indices = [
            idx for idx, is_valid in enumerate(sample["session_valid"].tolist())
            if is_valid and sample["sessions"][idx] is not None
        ]
        if len(valid_indices) <= 1 or np.random.random() >= self.session_drop_prob:
            return sample  # 不满足dropout条件，直接返回

        # 随机选择一个有效会话进行丢弃
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
                    pool.imap(self._load_participant_safe, range(n)),
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
                    self._cache[i] = self._load_participant(i)
                except Exception as exc:
                    errors += 1
                    if errors <= 3:
                        log.warning(f"Preload: participant {i} failed: {exc}")

        if errors > 0:
            log.warning(f"Preload: {errors}/{n} participants failed")
        gb = self._estimate_cache_bytes() / 1024**3
        log.info(f"Preloaded {n - errors}/{n} participants ({gb:.1f} GB in RAM)")
        return gb

    def _load_participant_safe(self, idx: int) -> dict[str, Any] | None:
        """多进程安全的参与者加载（捕获异常）"""
        try:
            return self._load_participant(idx)
        except Exception as exc:
            log.warning(f"Preload: participant {idx} failed: {exc}")
            return None

    def _estimate_cache_bytes(self) -> int:
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

    @property
    def is_preloaded(self) -> bool:
        return self._cache is not None


def grouped_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    批处理函数：将多个参与者的数据打包成批次

    核心挑战：
    1. 每个参与者有不同数量的有效会话（1-4个）
    2. 每个会话的序列长度不同

    解决方案：
    1. 将所有会话"展平"成一维列表（flat_batch）
    2. 通过session_types和session_valid追踪每个会话属于哪个参与者
    3. 缺失的会话用dummy填充，保持结构一致

    示例：
    输入：2个参与者
    - P1: [A01✓, B01✓, B02✗, B03✓]  (3个有效会话)
    - P2: [A01✓, B01✗, B02✓, B03✓]  (3个有效会话)

    输出：flat_batch包含8个会话（6个真实+2个dummy）
    """
    B = len(batch)
    all_sessions = []  # 展平后的所有会话数据
    session_types = []  # 每个会话的类型索引：0=A01, 1=B01, 2=B02, 3=B03
    session_valid_list = []
    flat_pids = []  # 每个会话对应的参与者ID
    flat_sess_names = []  # 每个会话的名称

    for sample in batch:
        session_valid_list.append(sample["session_valid"])
        for s_idx, sess_data in enumerate(sample["sessions"]):
            if sess_data is not None:
                # 有效会话：直接添加
                all_sessions.append(sess_data)
                session_types.append(s_idx)
                flat_pids.append(sample["anon_pid"])
                flat_sess_names.append(SESSIONS[s_idx])
            else:
                # 缺失会话：创建dummy占位
                # 步骤1：寻找参考会话（用于获取特征维度）
                ref = None
                for s in sample["sessions"]:  # 优先从当前参与者找
                    if s is not None:
                        ref = s
                        break
                if ref is None:  # 当前参与者所有会话都缺失，从其他参与者找
                    for other in batch:
                        for s in other["sessions"]:
                            if s is not None:
                                ref = s
                                break
                        if ref is not None:
                            break

                if ref is not None:
                    dummy = _make_dummy_session(ref)
                    all_sessions.append(dummy)
                    session_types.append(s_idx)
                    flat_pids.append(sample["anon_pid"])
                    flat_sess_names.append(SESSIONS[s_idx])

    if not all_sessions:
        log.warning("No valid sessions in batch — returning None (batch will be skipped)")
        return None

    # 展平后的批次大小和最大序列长度
    n_flat = len(all_sessions)
    T_max = max(s["seq_len"] for s in all_sessions)

    audio_names = list(all_sessions[0]["audio_groups"].keys())
    pooled_audio_names = list(all_sessions[0]["audio_pooled_groups"].keys())
    video_names = list(all_sessions[0]["video_groups"].keys())

    def _pad_groups(names, key):
        """填充序列特征组到(n_flat, T_max, D)"""
        result = {}
        for n in names:
            D = all_sessions[0][key][n].shape[-1]
            t = torch.zeros(n_flat, T_max, D)
            for i, s in enumerate(all_sessions):
                L = s["seq_len"]
                t[i, :L] = s[key][n]
            result[n] = t
        return result

    def _pad_1d(key, dtype=torch.float32):
        """填充一维序列到(n_flat, T_max)"""
        t = torch.zeros(n_flat, T_max, dtype=dtype)
        for i, s in enumerate(all_sessions):
            L = s["seq_len"]
            t[i, :L] = s[key]
        return t

    # 生成填充掩码
    pad_mask = torch.ones(n_flat, T_max, dtype=torch.bool)
    for i, s in enumerate(all_sessions):
        pad_mask[i, :s["seq_len"]] = False

    # 构建展平批次（所有会话作为独立样本）
    flat_batch = {
        "audio_groups": _pad_groups(audio_names, "audio_groups"),
        "audio_pooled_groups": {
            name: torch.stack([s["audio_pooled_groups"][name] for s in all_sessions])
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
                [s["audio_pooled_present"].get(name, False) for s in all_sessions],
                dtype=torch.bool,
            )
            for name in pooled_audio_names
        },
        "session_idx": torch.tensor([s["session_idx"] for s in all_sessions], dtype=torch.long),
        "seq_len": torch.tensor([s["seq_len"] for s in all_sessions], dtype=torch.long),
        "anon_pid": flat_pids,
        "session": flat_sess_names,
    }

    # 返回两层结构：flat_batch（会话级）+ participant级元信息
    result = {
        "flat_batch": flat_batch,  # 展平的会话数据，形状(n_flat, ...)
        "participant_y_a1": torch.stack([b["y_a1"] for b in batch]),  # 参与者级标签，形状(B, 3)
        "participant_y_a2": torch.stack([b["y_a2"] for b in batch]),  # 形状(B, 21)
        "participant_aux_attrs": torch.stack([b["aux_attrs"] for b in batch]),  # 形状(B, 5)，辅助属性
        "session_valid": torch.from_numpy(np.stack(session_valid_list)),  # 形状(B, 4)，标记每个参与者的哪些会话有效
        "session_types": torch.tensor(session_types, dtype=torch.long),  # 形状(n_flat,)，每个会话的类型索引
        "n_participants": B,  # 批次中的参与者数量
        "anon_pids": [b["anon_pid"] for b in batch],
        "anon_schools": [b["anon_school"] for b in batch],
        "anon_classes": [b["anon_class"] for b in batch],
        "flat_sessions": flat_sess_names,  # 展平后的会话名称列表
        "flat_pids": flat_pids,  # 展平后的参与者ID列表
    }

    # 添加辅助任务标签
    if "auxiliary_targets" in batch[0]:
        result["auxiliary_targets"] = {
            "emotion_dims": torch.stack([b["auxiliary_targets"]["emotion_dims"] for b in batch]),
        }

    return result


def _make_dummy_session(ref: dict[str, Any]) -> dict[str, Any]:
    """
    创建零填充的dummy会话，用于占位缺失的会话

    参数：
        ref: 参考会话，用于获取特征维度和结构

    返回：
        与ref结构相同但数值全为零的dummy会话

    设计要点：
    - 序列长度设为1（最小长度），节省内存
    - 所有掩码为False，标记为无效数据
    - 模型通过session_valid识别dummy，不参与损失计算
    """
    T = 1  # 最小序列长度
    audio_groups = {k: torch.zeros(T, v.shape[-1]) for k, v in ref["audio_groups"].items()}
    video_groups = {k: torch.zeros(T, v.shape[-1]) for k, v in ref["video_groups"].items()}
    return {
        "audio_groups": audio_groups,
        "audio_pooled_groups": {
            k: torch.zeros_like(v) for k, v in ref["audio_pooled_groups"].items()
        },
        "video_groups": video_groups,
        "mask_audio": torch.zeros(T, dtype=torch.bool),  # 全False，标记为无效
        "mask_video": torch.zeros(T, dtype=torch.bool),
        "vad_signal": torch.zeros(T),
        "qc_quality": torch.zeros(T),
        "audio_pooled_present": {
            k: False for k in ref["audio_pooled_groups"].keys()
        },
        "session_idx": 0,  # 默认为A01
        "seq_len": T,
        "session": "A01",
    }
