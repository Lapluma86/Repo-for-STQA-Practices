# MTL（多任务学习）集成指南

## 当前状态

✅ **已实现的模块**：
- `common/models/mtl_uncertainty.py` - 不确定性加权 + 辅助任务头
- `common/models/phase1_integration.py` - 集成包装器
- 配置文件已准备好（`tasks/a1/phase1_optimization.yaml`）

❌ **缺失的部分**：
1. **辅助任务标签数据** - 需要准备情绪维度、情感分类、AU标签
2. **数据加载器修改** - `GroupedParticipantDataset` 需要加载辅助标签
3. **训练循环集成** - `runner.py` 需要使用 `OptimizedGroupedModel`

---

## 方案选择

### 方案A：仅使用优化损失函数（推荐，立即可用）

**优点**：
- 无需修改代码，只需配置
- 数据已存在（辅助属性在CSV中）
- 风险低，预期提升 +3~8%

**配置**：
```yaml
# 已在 tasks/a1/phase1_optimization.yaml 中配置
use_combined_loss: 1      # ASL + Soft-F1
use_aux_attrs: true       # 使用辅助属性
enable_auxiliary_tasks: false  # 不启用辅助任务
```

**运行**：
```bash
python train.py --task a1 --config tasks/a1/phase1_optimization.yaml
```

---

### 方案B：完整MTL（需要准备辅助标签）

**优点**：
- 预期提升更大 +5~10%
- 充分利用多任务学习

**缺点**：
- 需要准备辅助标签数据
- 需要修改代码集成

---

## 方案B实施步骤

### 步骤1：准备辅助任务标签

修改 `common/data/grouped_dataset.py`，添加辅助标签加载函数：

```python
def _load_emotion_dims(self, participant_info: dict) -> np.ndarray:
    """
    从DASS-21分数推导情绪维度
    
    返回: (2,) [valence, arousal]
    - valence（愉悦度）：抑郁分数越高，valence越低
    - arousal（激活度）：焦虑分数越高，arousal越高
    """
    y_a1 = participant_info["y_a1"]  # [depression, anxiety, stress]
    depression = y_a1[0]
    anxiety = y_a1[1]
    
    # 归一化到 [0, 1]
    valence = 1.0 - (depression / 3.0)  # 抑郁越高，愉悦度越低
    arousal = anxiety / 3.0              # 焦虑越高，激活度越高
    
    return np.array([valence, arousal], dtype=np.float32)


def _load_emotion_cls(self, participant_info: dict) -> int:
    """
    从DASS-21分数推导情感分类
    
    返回: 0=快乐, 1=悲伤, 2=愤怒, 3=中性
    """
    y_a1 = participant_info["y_a1"]
    depression, anxiety, stress = y_a1
    
    # 简单规则：
    if depression > 1.5:
        return 1  # 悲伤
    elif stress > 1.5:
        return 2  # 愤怒
    elif depression < 0.5 and anxiety < 0.5:
        return 0  # 快乐
    else:
        return 3  # 中性


def _load_au_labels(self, sess_row: pd.Series) -> np.ndarray:
    """
    从OpenFace特征提取AU标签
    
    返回: (12,) 12个关键AU的激活强度 [0-5]
    """
    # 假设OpenFace特征已经包含AU强度
    # 这里需要根据实际特征文件格式实现
    # 示例：从视频特征中提取AU_01_r, AU_02_r, ..., AU_45_r
    
    # 关键AU列表（根据情绪识别文献选择）
    key_aus = [1, 2, 4, 5, 6, 7, 9, 12, 15, 17, 20, 25]
    
    # TODO: 实际实现需要读取OpenFace输出文件
    # 这里返回占位符
    return np.zeros(12, dtype=np.float32)
```

在 `__getitem__` 方法中添加辅助标签：

```python
def __getitem__(self, idx: int) -> dict:
    info = self.participants[idx]
    
    # ... 现有代码加载特征 ...
    
    sample = {
        "features": features,
        "masks": masks,
        "session_valid": session_valid,
        "session_types": session_types,
        "y_a1": info["y_a1"],
        "y_a2": info["y_a2"],
        "aux_attrs": info["aux_attrs"],
    }
    
    # 添加辅助任务标签
    if self.split == "train":  # 仅训练集需要
        sample["auxiliary_targets"] = {
            "emotion_dims": self._load_emotion_dims(info),
            "emotion_cls": self._load_emotion_cls(info),
            "au_labels": self._load_au_labels(info["sess_rows"]["A01"]),  # 使用第一个会话
        }
    
    return sample
```

---

### 步骤2：修改训练循环

修改 `common/runner.py` 中的训练函数：

```python
def train_epoch_with_mtl(
    optimized_model: OptimizedGroupedModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    task: str,
    epoch: int,
    epochs: int,
    # ... 其他参数 ...
) -> float:
    """使用MTL的训练循环"""
    optimized_model.train()
    total_loss = 0.0
    n_batches = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}")
    for batch in pbar:
        # 准备数据
        flat_batch = {k: v.to(device) for k, v in batch["features"].items()}
        targets = {
            "participant_y": batch["y_a1" if task == "a1" else "y_a2"].to(device),
            "session_types": batch["session_types"].to(device),
        }
        
        # 添加辅助任务标签
        if "auxiliary_targets" in batch:
            targets["auxiliary_targets"] = {
                k: v.to(device) for k, v in batch["auxiliary_targets"].items()
            }
        
        aux_attrs = batch.get("aux_attrs")
        if aux_attrs is not None:
            aux_attrs = aux_attrs.to(device)
        
        # 前向传播
        outputs = optimized_model(
            flat_batch,
            n_participants=len(batch["y_a1"]),
            session_valid=batch["session_valid"].to(device),
            aux_attrs=aux_attrs,
        )
        
        # 计算损失（使用phase1_integration.py中的函数）
        from common.models.phase1_integration import compute_optimized_loss
        
        loss, loss_dict = compute_optimized_loss(
            outputs=outputs,
            targets=targets,
            model=optimized_model,
            task=task,
            session_valid=batch["session_valid"].to(device),
            # ... 传递所有损失参数 ...
        )
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(optimized_model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
        
        # 显示详细损失
        pbar.set_postfix(loss_dict)
    
    return total_loss / max(n_batches, 1)
```

---

### 步骤3：修改主训练函数

在 `common/runner.py` 的 `main()` 函数中：

```python
def main():
    # ... 现有代码 ...
    
    # 检查是否启用MTL
    enable_mtl = cfg.get("enable_auxiliary_tasks", False)
    
    if enable_mtl:
        # 使用优化模型
        from common.models.phase1_integration import OptimizedGroupedModel
        
        optimized_model = OptimizedGroupedModel(
            grouped_model=grouped_model,
            participant_head=participant_head,
            session_head=session_head,
            d_shared=cfg["d_shared"],
            aux_dim=aux_encoder.output_dim if use_aux_attrs else 0,
            use_uncertainty_weighting=cfg.get("use_uncertainty_weighting", True),
            enable_auxiliary_tasks=True,
            enable_emotion_dims=cfg.get("enable_emotion_dims", True),
            enable_emotion_cls=cfg.get("enable_emotion_cls", True),
            enable_au_pred=cfg.get("enable_au_pred", False),
        ).to(device)
        
        # 使用MTL训练循环
        for epoch in range(1, epochs + 1):
            train_loss = train_epoch_with_mtl(
                optimized_model, train_loader, optimizer, device, task, epoch, epochs,
                # ... 参数 ...
            )
    else:
        # 使用原有训练循环（当前实现）
        for epoch in range(1, epochs + 1):
            train_loss = train_epoch_grouped(
                grouped_model, participant_head, session_head,
                train_loader, optimizer, device, task, epoch, epochs,
                # ... 参数 ...
            )
```

---

### 步骤4：更新配置文件

修改 `tasks/a1/phase1_optimization.yaml`：

```yaml
# 启用完整MTL
use_uncertainty_weighting: true
enable_auxiliary_tasks: true
enable_emotion_dims: true
enable_emotion_cls: true
enable_au_pred: false  # AU预测需要额外处理，暂时禁用
```

---

## 验证步骤

### 1. 测试辅助标签加载

```python
from common.data.grouped_dataset import GroupedParticipantDataset
from common.data.dataset import FeatureConfig

ds = GroupedParticipantDataset('manifests/train.csv', FeatureConfig(), 'train')
sample = ds[0]

print("Auxiliary targets keys:", sample.get("auxiliary_targets", {}).keys())
print("Emotion dims shape:", sample["auxiliary_targets"]["emotion_dims"].shape)
print("Emotion cls:", sample["auxiliary_targets"]["emotion_cls"])
```

### 2. 测试模型前向传播

```bash
python test_phase1_optimization.py
```

### 3. 运行训练

```bash
python train.py --task a1 --config tasks/a1/phase1_optimization.yaml --epochs 2
```

观察日志中是否出现：
- `main_loss`
- `session_loss`
- `session_type_loss`
- `aux_emotion_dims_loss`
- `aux_emotion_cls_loss`
- `task_weight_0`, `task_weight_1`, ... （不确定性权重）

---

## 预期效果

### 方案A（仅优化损失）
- A1 F1: +3~8%
- A2 QWK: +3~5%
- 训练时间：无明显增加

### 方案B（完整MTL）
- A1 F1: +5~10%
- A2 QWK: +5~8%
- 训练时间：+10~20%（额外的辅助任务计算）

---

## 常见问题

### Q1: 为什么辅助属性和辅助任务不一样？

**辅助属性**（Auxiliary Attributes）：
- 静态背景信息（家庭结构、独生子女等）
- 数据已在CSV中，直接可用
- 通过embedding编码后拼接到表示向量

**辅助任务**（Auxiliary Tasks）：
- 额外的预测任务（情绪维度、情感分类、AU预测）
- 需要准备标签数据
- 通过多任务学习共享表示

### Q2: 不确定性加权是什么？

自动学习每个任务的权重，避免手动调参：
```
总损失 = Σ (1/2σ²) × L_i + log(σ_i)
```
- σ²大 → 任务不确定性高 → 权重小
- σ²小 → 任务确定性高 → 权重大

### Q3: 如果没有AU标签怎么办？

可以只启用情绪维度和情感分类：
```yaml
enable_emotion_dims: true
enable_emotion_cls: true
enable_au_pred: false
```

---

## 总结

**立即可用**：方案A，只需运行：
```bash
python train.py --task a1 --config tasks/a1/phase1_optimization.yaml
```

**完整MTL**：方案B，需要完成3个步骤：
1. 修改 `grouped_dataset.py` 加载辅助标签
2. 修改 `runner.py` 集成 `OptimizedGroupedModel`
3. 更新配置启用MTL

建议先用方案A验证优化损失的效果，再考虑实施方案B。
