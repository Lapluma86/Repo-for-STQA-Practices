"""
特征文件I/O模块

本模块是ADODAS2026比赛baseline的数据加载基础模块，负责从磁盘读取各种类型的特征文件。
支持两种特征类型：
1. 序列特征（sequence.npz）：时序数据，包含features、timestamps_ms、valid_mask
2. 池化特征（pooled.parquet/json）：全局统计特征，如egemaps

目录结构：
<feature_root>/<split>/<anon_school>/<anon_class>/<anon_pid>/<modality>/<feature_set>/[<model_tag>]/<session>/
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import numpy as np


class SequenceData(NamedTuple):
    """
    时序特征数据的容器类

    使用NamedTuple的原因：
    1. 不可变，防止意外修改数据
    2. 支持字段访问（seq.features），比普通tuple更直观
    3. 轻量级，无需实例化开销

    Attributes:
        features: 特征矩阵，形状(T, D)，T为时间帧数，D为特征维度
        timestamps_ms: 时间戳数组，形状(T)，单位毫秒，用于多模态对齐
        valid_mask: 有效帧掩码，形状(T)，标记因设备故障、遮挡等原因导致的无效帧
    """
    features: np.ndarray
    timestamps_ms: np.ndarray
    valid_mask: np.ndarray     


_MEL_MFCC_KEYS = ("mel_features", "mfcc_features")
_GENERIC_KEY = "features"


def load_sequence(
    root: Path,
    split: str,
    anon_school: str,
    anon_class: str,
    anon_pid: str,
    modality: str,
    feature_set: str,
    session: str,
    model_tag: str | None = None,
) -> SequenceData:
    parts = [root, split, anon_school, anon_class, anon_pid, modality, feature_set] # 构建路径
    if model_tag is not None:
        parts.append(model_tag) # SSL特征需要多一层目录
    parts.append(session)
    seq_path = Path(*[str(p) for p in parts]) / "sequence.npz"

    if not seq_path.exists():
        raise FileNotFoundError(f"Missing sequence file: {seq_path}") # 是否存在目录

    data = np.load(str(seq_path), allow_pickle=True) # 加载npz文件，npz是numpy压缩格式，可存储多个数组

    if feature_set == "mel_mfcc":
        arrays = []
        for k in _MEL_MFCC_KEYS:
            if k not in data:
                raise KeyError(f"Expected key '{k}' in {seq_path}, found {list(data.keys())}")
            arrays.append(data[k].astype(np.float32))
        features = np.concatenate(arrays, axis=-1) # mel_mfcc包含两个独立数组，需要进行拼接，沿最后一维拼接
    # 何为mel_mfcc特征？mel频谱图和MFCC系数的组合特征，常用于语音分析，mel频谱图捕捉频谱能量分布，MFCC捕捉语音的短时特征
    elif _GENERIC_KEY in data:
        features = data[_GENERIC_KEY].astype(np.float32) # 对于其他特征集，直接使用"features"键存储特征矩阵
    else:
        raise KeyError(
            f"No known feature key in {seq_path}. Keys: {list(data.keys())}"
        )

    if features.ndim == 1:
        features = features[:, np.newaxis] # 确保特征矩阵至少是二维的，如果是一维则添加一个维度，变成(T, 1)

    if "timestamps_ms" not in data:
        raise KeyError(f"Missing 'timestamps_ms' in {seq_path}")
    timestamps_ms = data["timestamps_ms"].astype(np.float64) # 时间戳通常以毫秒为单位，使用float64以保持精度

    if "valid_mask" in data:
        valid_mask = data["valid_mask"].astype(bool)
    else:
        valid_mask = np.ones(len(timestamps_ms), dtype=bool) # 如果没有valid_mask，默认所有帧都是有效的

    T = len(timestamps_ms)
    if features.shape[0] != T:
        raise ValueError(
            f"Shape mismatch in {seq_path}: features {features.shape[0]} vs timestamps {T}"
        )
    if valid_mask.shape[0] != T:
        raise ValueError(
            f"Shape mismatch in {seq_path}: valid_mask {valid_mask.shape[0]} vs timestamps {T}"
        ) # 这一段的作用是确保三个数组的时间维度一致，防止后续处理时出现对齐问题

    return SequenceData(features=features, timestamps_ms=timestamps_ms, valid_mask=valid_mask)


def load_egemaps_pooled(
    root: Path,
    split: str,
    anon_school: str,
    anon_class: str,
    anon_pid: str,
    session: str,
) -> np.ndarray | None: # egemaps是一种常用的语音特征集，包含多种统计特征，如基频、能量、MFCC等的均值、方差等统计量，适合用于情感分析等任务,没有时序特征，只有池化特征
    base = root / split / anon_school / anon_class / anon_pid / "audio" / "egemaps" / session

    parquet_path = base / "pooled.parquet"
    if parquet_path.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(parquet_path)
            return df.iloc[0].values.astype(np.float32) # 如果存在pooled.parquet文件，使用pandas读取parquet格式的文件，parquet是一种列式存储格式，适合存储表格数据，读取后取第一行的值作为特征向量，并转换为float32类型
        except Exception:
            pass
        try:
            import pandas as pd
            df = pd.read_parquet(parquet_path, engine="fastparquet") # 有些环境可能默认使用pyarrow引擎读取parquet文件，如果遇到问题，可以尝试指定使用fastparquet引擎，fastparquet是另一个流行的parquet读取库，可能在某些环境下更兼容
            return df.iloc[0].values.astype(np.float32) 
        except Exception:
            pass

    json_path = base / "pooled.json" # 若parquet不存在，则尝试读取json文件，json是一种轻量级的数据交换格式，适合存储简单的键值对数据，如果存在pooled.json文件，使用json模块读取，期望其中有一个"features"键，其值是一个字典，包含特征名称和对应的数值，将这些数值提取出来并转换为float32类型的numpy数组返回
    if json_path.exists():
        try:
            with open(json_path) as f:
                meta = json.load(f) # 解析JSON文件，得到一个Python字典对象
            if "features" in meta and isinstance(meta["features"], dict): # 检查是否存在"features"键，并且其值是一个字典，符合预期的格式
                vals = np.array(list(meta["features"].values()), dtype=np.float32) # 提取特征值，转换为numpy数组，并指定数据类型为float32
                return vals
        except Exception:
            pass

    return None


def discover_feature_sets( # 此函数用于发现指定分割和模态下的特征集名称以及对应的模型标签
    root: Path, split: str, modality: str, limit: int = 5
) -> dict[str, list[str]]:
    split_dir = root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    result: dict[str, list[str]] = {}
    count = 0
    for sch in sorted(split_dir.iterdir()):
        if not sch.is_dir():
            continue
        for cls_ in sorted(sch.iterdir()):
            if not cls_.is_dir():
                continue
            for pid in sorted(cls_.iterdir()):
                if not pid.is_dir():
                    continue # 逐步遍历目录结构，找到对应的模态目录，检查其中的特征集目录，收集特征集名称和对应的模型标签（如果有的话），直到达到指定的数量限制
                mod_dir = pid / modality
                if not mod_dir.exists():
                    continue
                for feat in sorted(mod_dir.iterdir()):
                    if not feat.is_dir():
                        continue
                    name = feat.name # 例如这里的name就可能是"egemaps"或者"mel_mfcc"，即特征集的名称
                    if name not in result:
                        sub_dirs = [d.name for d in sorted(feat.iterdir()) if d.is_dir()] # 获取特征目录下子目录名称
                        sessions = {"A01", "B01", "B02", "B03"}
                        model_tags = [s for s in sub_dirs if s not in sessions] # 过滤掉会话目录，剩下的就是模型标签目录，如果存在的话
                        if model_tags:
                            result[name] = sorted(model_tags)
                        else:
                            result[name] = []
                count += 1
                if count >= limit:
                    return result
    return result


def list_file_ids(root: Path, split: str, limit: int = 0) -> list[tuple[str, str, str]]:
    split_dir = root / split
    results: list[tuple[str, str, str]] = []
    for sch in sorted(split_dir.iterdir()):
        if not sch.is_dir():
            continue
        for cls_ in sorted(sch.iterdir()):
            if not cls_.is_dir():
                continue
            for pid in sorted(cls_.iterdir()):
                if not pid.is_dir():
                    continue
                results.append((sch.name, cls_.name, pid.name)) # 不必多说
                if limit > 0 and len(results) >= limit:
                    return results
    return results
