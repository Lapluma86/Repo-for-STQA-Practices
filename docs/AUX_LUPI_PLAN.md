# 辅助属性利用研究方案（AdoDAS 2026）

> **本文档面向 Claude Code。** 它定义了基于训练时辅助属性（家庭结构、独生子女、父母偏爱、成绩变动、情绪变动）提升测试集性能的实施计划。所有改动以 baseline（`common/runner.py` + `common/models/grouped_model.py` + `tasks/{a1,a2}/default.yaml` 现状）为基准，按阶段独立交付，每阶段都应可单独训练并产生可验证的指标。

---

## 0. 背景与目标

### 0.1 任务范式

辅助属性仅在训练集存在，在 `test_hidden` 中不可用。这是经典的 **Learning Using Privileged Information (LUPI)** 设定。所有改动必须满足以下硬约束：

1. 推理（`infer.py`）的前向路径不依赖任何辅助属性。
2. 测试时 checkpoint 在不传入辅助属性的情况下能正常推理，输出格式与 baseline 完全一致（`p_D,p_A,p_S` 或 `d01..d21`）。
3. 任何对辅助属性的使用必须可通过 YAML 配置整体关闭，回退到 baseline 行为。

### 0.2 五个辅助属性的字段定义

| 字段名（约定） | 类别数 | 含义 | 缺失语义 |
|--------------|-------|------|---------|
| `aux_family` | 6 | 家庭结构（核心/大/单亲/重组/隔代/其他） | 无缺失 |
| `aux_only_child` | 2 | 是否独生子女 | 无缺失 |
| `aux_favoritism` | 3 | 父母偏爱感知 | 独生子女时**结构性缺失** |
| `aux_academic` | 3 | 学习成绩变动 | 可能缺失 |
| `aux_emotional` | 3 | 情绪状态变动 | 可能缺失 |

所有属性需读入为整数索引（0-indexed），缺失统一标记为 `-1`。

### 0.3 性能目标（相对 baseline）

| 阶段 | 改动 | 预期增益（验证集主指标） |
|------|------|----------------------|
| Phase 1 | 多任务辅助监督头 | A1 +1~3% F1 / A2 +1~2% QWK |
| Phase 2 | 辅助属性驱动的样本加权 | A1 +0.5~1.5% F1 / A2 +0.5~1% QWK |
| Phase 3 | 特权知识蒸馏 | A1 +1~2% F1 / A2 +1% QWK |
| Phase 4 | Group DRO（可选） | A1 +0.5~1% F1（子群均衡） |

每个阶段必须比上一阶段在验证集**主指标**上有正向提升，否则**回滚该阶段配置**并记录失败原因。

---

## 1. 总体设计原则

### 1.1 配置驱动

所有新增功能挂在 YAML 配置下，默认值与 baseline 行为一致。任何阶段都可通过设置 `aux_lupi.enabled: false` 完全关闭。新增的顶层 YAML 节点统一放在 `aux_lupi:` 之下：

```yaml
aux_lupi:
  enabled: true              # 总开关
  attribute_path: data/aux_attributes.csv  # 辅助属性表路径
  phase1_mtl:                # Phase 1 多任务监督
    enabled: true
    weights:
      family: 0.05
      only_child: 0.05
      favoritism: 0.05
      academic: 0.15
      emotional: 0.20
    dropout_for_distillation: 0.0  # Phase 3 用，Phase 1 设 0
  phase2_reweight:           # Phase 2 样本加权
    enabled: false
    method: emotional_consistency  # emotional_consistency / academic_consistency / joint
    weight_low: 0.7
    weight_high: 1.2
  phase3_kd:                 # Phase 3 知识蒸馏
    enabled: false
    teacher_checkpoint: null  # 必须先训练 teacher 并填入路径
    temperature: 2.5
    alpha: 0.4
  phase4_dro:                # Phase 4 Group DRO
    enabled: false
    grouping_attrs: [aux_academic, aux_emotional]
    eta: 0.5
```

### 1.2 单点改动原则

每个阶段最多触碰 3~5 个文件。Phase 1 是基础，Phase 2/3/4 都建立在 Phase 1 的数据加载之上。

### 1.3 检查点兼容性

模型权重的 state_dict 必须保持 **baseline 推理代码可加载**：辅助任务头作为可选模块，加载时如果 state_dict 缺失对应 key 应跳过，而不是报错。

### 1.4 与 baseline 已有机制的关系

baseline 已有 `SessionTypeClassifier`（`common/models/grouped_model.py:56-66`）作为辅助任务，损失通过 `session_type_loss_weight` 加权（A1=0.2, A2=0.5）。**新增的辅助任务监督完全复用这种模式**——同样的位置、同样的损失加权方式。不要发明新机制。

---

## 2. Phase 1：多任务辅助监督头（必做）

### 2.1 目的

让 `participant_repr (B, 256)` 在预测 DASS 之外额外预测 5 个辅助属性。共享表示被迫编码与心理状态相关的潜变量，提升泛化。

### 2.2 文件改动清单

| 文件 | 改动类型 | 说明 |
|------|---------|------|
| `common/data/grouped_dataset.py` | 修改 | 读取辅助属性并附加到 batch |
| `common/data/grouped_collate.py`（或对应 collate 函数所在文件） | 修改 | 把辅助属性 stack 成 tensor |
| `common/models/heads.py` | 新增类 | `AuxAttributeHeads` |
| `common/models/grouped_model.py` | 修改 | 在 `GroupedModel.__init__` 和 `forward` 中挂载辅助头 |
| `common/runner.py` | 修改 | 损失合成、metric 记录、checkpoint 处理 |
| `tasks/a1/default.yaml` | 修改 | 添加 `aux_lupi` 配置块 |
| `tasks/a2/default.yaml` | 修改 | 添加 `aux_lupi` 配置块 |
| `infer.py` | 修改（轻微） | 加载 checkpoint 时容忍辅助头缺失/存在 |

### 2.3 数据层改动（`grouped_dataset.py`）

辅助属性以单独 CSV 形式提供，按 `(anon_school, anon_class, anon_pid)` 索引。Manifest CSV 已经有这三列。在 `GroupedParticipantDataset.__init__` 中：

1. 读取 `config.aux_lupi.attribute_path` 指向的 CSV。
2. 用 `(anon_school, anon_class, anon_pid)` 做 key 构建 dict。
3. 在 `__getitem__` 返回的样本字典中添加键 `aux_attrs: Dict[str, int]`（5 个字段，缺失为 -1）。
4. 如果 `aux_lupi.enabled = false`，跳过整个加载逻辑，返回的样本字典中不包含 `aux_attrs` 键，下游通过 `if "aux_attrs" in batch` 判断。

注意 **test_hidden split 不应加载辅助属性**——即使 CSV 中有，也不读，确保推理路径不依赖该文件存在。判断逻辑：`split != "test_hidden"`。

### 2.4 Collate 函数改动

在 `grouped_collate_fn` 中：

```python
if "aux_attrs" in samples[0]:
    batch["aux_attrs"] = {
        name: torch.tensor([s["aux_attrs"][name] for s in samples], dtype=torch.long)
        for name in ["aux_family", "aux_only_child", "aux_favoritism", "aux_academic", "aux_emotional"]
    }
```

### 2.5 模型改动（`heads.py` 新增类）

在 `common/models/heads.py` 添加：

```python
class AuxAttributeHeads(nn.Module):
    """五个辅助属性的并行分类头。与 SessionTypeClassifier 同构。"""
    NUM_CLASSES = {
        "aux_family": 6,
        "aux_only_child": 2,
        "aux_favoritism": 3,
        "aux_academic": 3,
        "aux_emotional": 3,
    }

    def __init__(self, d_shared: int = 256, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(d_shared, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, n_cls),
            )
            for name, n_cls in self.NUM_CLASSES.items()
        })

    def forward(self, participant_repr: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {name: head(participant_repr) for name, head in self.heads.items()}
```

### 2.6 GroupedModel 改动

在 `common/models/grouped_model.py`：

1. `__init__` 中根据 `config.aux_lupi.enabled and config.aux_lupi.phase1_mtl.enabled` 决定是否构造 `self.aux_heads = AuxAttributeHeads(d_shared)`。否则置为 `None`。
2. `forward` 返回字典中添加 `aux_logits: Optional[Dict[str, Tensor]]`，仅当 `self.aux_heads is not None` 时计算，否则置 None。

**关键：辅助头从 `participant_repr` 派生，不影响 session-level 流程**。

### 2.7 损失合成（`runner.py`）

在主训练循环里，参考现有的 `session_type_loss` 处理方式，在 main loss 后追加：

```python
aux_loss = torch.tensor(0.0, device=device)
if model.aux_heads is not None and "aux_attrs" in batch:
    aux_logits = outputs["aux_logits"]  # Dict[name -> (B, C)]
    aux_weights = config.aux_lupi.phase1_mtl.weights
    for name, logits in aux_logits.items():
        targets = batch["aux_attrs"][name].to(device)  # (B,)
        valid_mask = targets >= 0  # -1 表示缺失
        if valid_mask.sum() == 0:
            continue
        ce = F.cross_entropy(logits[valid_mask], targets[valid_mask], reduction="mean")
        aux_loss = aux_loss + aux_weights[name.replace("aux_", "")] * ce

total_loss = main_loss + session_loss_weight * sess_loss \
           + session_type_loss_weight * type_loss \
           + aux_loss  # 已含权重
```

**注意 mask 处理**：`aux_favoritism` 对独生子女结构性缺失（CSV 中应是 -1），必须用 `valid_mask` 跳过这些样本，否则 cross_entropy 会因为 target=-1 报错。

### 2.8 YAML 配置改动

在 `tasks/a1/default.yaml` 末尾添加 1.1 节给出的 `aux_lupi` 块。`tasks/a2/default.yaml` 同理。Phase 1 应保持 `phase2_reweight.enabled = false`、`phase3_kd.enabled = false`、`phase4_dro.enabled = false`。

### 2.9 推理代码兼容性（`infer.py`）

在加载 checkpoint 时使用 `strict=False`：

```python
model.load_state_dict(ckpt["model"], strict=False)
```

并显式记录 missing/unexpected keys 到日志，确认只有辅助头相关的 keys 出现在 unexpected 中。

### 2.10 Phase 1 验证准则（Definition of Done）

1. 启用 `aux_lupi.enabled=true, phase1_mtl.enabled=true`，训练能正常完成，无 NaN、无 OOM。
2. 关闭 `aux_lupi.enabled=false`，训练结果应与原 baseline **完全一致**（同样的 seed → 同样的 best metric，误差 < 1e-4）。这是回退安全性的关键测试。
3. 训练日志中应能看到每个辅助任务的训练 loss 和验证准确率。如果某个辅助任务的验证准确率始终接近随机（如 `aux_family` 始终 ~16.7%），说明该任务过难或标签与表示不可学，应在配置中将其权重降到 0.01。
4. 验证集主指标（A1: macro F1 / A2: mean QWK）相对 baseline 应有提升。若无提升或下降，按 2.11 进行调试。
5. 使用 baseline 的 `infer.py` 加载新 checkpoint 能正常推理且输出格式正确。

### 2.11 Phase 1 调试 checklist

- **辅助 loss 不下降**：检查辅助属性的标签是否正确加载（打印一个 batch 看 `aux_attrs` 字典内容）；检查 `valid_mask` 是否正确处理 -1。
- **主任务变差**：辅助权重过高。把所有权重整体除以 2 重试。
- **训练发散**：检查辅助头的初始化（应用默认 PyTorch 初始化，不要额外加 bias init）；检查梯度是否被辅助 loss 主导（用 `loss.backward()` 后看 `participant_repr` 处的梯度范数）。
- **关闭后 baseline 行为变了**：说明改动有副作用，回到代码 diff 检查是否在 `enabled=false` 分支下有意外执行的代码。

---

## 3. Phase 2：辅助属性驱动的样本加权（去噪）

### 3.1 目的

DASS-21 是自评量表，存在 ~15% 标签噪声。`aux_emotional`（情绪状态变动）和 `aux_academic`（成绩变动）是 DASS 标签的部分独立观测。利用一致性识别可能错标的样本并降权，提升类别不平衡场景下的有效信噪比。

### 3.2 依赖

依赖 Phase 1 已完成的数据加载（`aux_attrs` 在 batch 中可用）。Phase 2 可以单独启用，也可与 Phase 1 同时启用（推荐组合）。

### 3.3 文件改动清单

| 文件 | 改动 |
|------|------|
| `common/runner.py` | 在 loss 计算前根据辅助属性计算 `sample_weight` |
| `tasks/{a1,a2}/default.yaml` | `phase2_reweight` 配置启用 |

### 3.4 一致性评分逻辑（`runner.py` 新增辅助函数）

```python
def compute_aux_consistency_weight(batch, labels, method="emotional_consistency",
                                    w_low=0.7, w_high=1.2, w_mid=1.0):
    """
    返回 (B,) 的 sample_weight tensor。
    - labels: A1 是 (B, 3), 使用 y_D 与情绪变动比；A2 用 d_avg = mean(d01..d21) 二值化。
    - 一致：DASS 阳性 + 情绪"变差"，或 DASS 阴性 + 情绪"变好" → w_high
    - 冲突：DASS 阳性 + 情绪"变好"，或 DASS 阴性 + 情绪"变差" → w_low
    - 缺失或中性："无变化" 或 aux_emotional=-1 → w_mid
    """
    weights = torch.full((labels.shape[0],), w_mid, device=labels.device)
    if "aux_attrs" not in batch:
        return weights

    aux_emo = batch["aux_attrs"]["aux_emotional"].to(labels.device)  # 0=better, 1=worse, 2=no_change, -1=missing

    # 对 A1：标签为 (B, 3)，看是否有任一标签为阳性
    # 对 A2：用 d01..d21 平均值 > 1.0 作为整体阳性指标
    if labels.dim() == 2 and labels.shape[1] == 3:        # A1
        label_pos = (labels.sum(dim=1) > 0)
    else:                                                  # A2
        label_pos = (labels.float().mean(dim=1) > 1.0)

    # "情绪变差" = 1，"情绪变好" = 0
    aux_worse = (aux_emo == 1)
    aux_better = (aux_emo == 0)

    consistent = (label_pos & aux_worse) | ((~label_pos) & aux_better)
    conflict = (label_pos & aux_better) | ((~label_pos) & aux_worse)

    weights[consistent] = w_high
    weights[conflict] = w_low
    return weights
```

### 3.5 与 baseline pos_weight 的关系

baseline 已经有 `pos_weight` 处理类别不平衡。**Phase 2 的样本加权不替换 pos_weight，而是乘上它**：

```python
# 主任务 loss 计算（A1 示例）
per_sample_loss = bce_loss_with_pos_weight(logits, targets)  # (B,) or (B, 3)
sample_weight = compute_aux_consistency_weight(batch, targets, ...)  # (B,)
if per_sample_loss.dim() == 2:
    sample_weight = sample_weight.unsqueeze(1)
main_loss = (per_sample_loss * sample_weight).mean()
```

需要把 baseline 中 `bce_loss(..., reduction='mean')` 改成 `reduction='none'`，然后手动加权平均。如果改动困难，可以等价地直接传 `weight` 参数给 BCEWithLogitsLoss（不是 pos_weight）。

### 3.6 Phase 2 验证准则

1. 启用 `phase2_reweight.enabled=true`，训练能正常完成。
2. 训练日志记录每个 epoch 中 high/mid/low 权重样本的比例（应大致为 25% / 60% / 15%；如果 low 权重占 > 30%，说明阈值定义有问题）。
3. 验证集主指标相对 Phase 1 应有正向变化（或至少持平）。如果显著下降，说明辅助属性与标签的相关性比预期弱，应将 `w_low` 调回 0.85（更保守）或关闭此阶段。
4. 重点观察少数类（A1 的阳性 F1，A2 的高分阈值阳性率）的变化，这是 Phase 2 主要优化的方向。

---

## 4. Phase 3：特权知识蒸馏

### 4.1 目的

训练一个**带辅助属性输入**的 Teacher，再训练一个**只用音视频**的 Student 匹配 Teacher 的软标签。Student 学到的是 Teacher 经过辅助属性条件化后的精细判断，但推理时不需要辅助属性。

### 4.2 训练流程

两阶段：

```
Stage A: 训练 Teacher
   - Teacher 模型 = baseline + 辅助属性 one-hot 拼接到 participant_repr
   - 主损失 + Phase 1 辅助监督 + Phase 2 样本加权
   - 输出 checkpoint: runs/teacher/<run_id>/checkpoints/best.pt

Stage B: 训练 Student
   - Student 模型 = baseline 结构（不接受辅助属性输入）
   - 主损失 + KD 损失（Student 软输出 vs Teacher 软输出）
   - 推理时只用 Student
```

### 4.3 Teacher 模型实现

新增文件 `common/models/teacher_model.py`：

```python
class TeacherModel(nn.Module):
    """与 GroupedModel 同构，但在 participant_repr 后拼接辅助属性 one-hot。"""

    AUX_DIMS = {  # 与 AuxAttributeHeads.NUM_CLASSES 一致
        "aux_family": 6, "aux_only_child": 2, "aux_favoritism": 3,
        "aux_academic": 3, "aux_emotional": 3,
    }
    AUX_TOTAL = sum(AUX_DIMS.values())  # 17

    def __init__(self, base_model: GroupedModel, d_shared: int = 256, aux_dropout: float = 0.2):
        super().__init__()
        self.base = base_model  # 共享 backbone + 聚合器
        # 拼接后维度 = d_shared + AUX_TOTAL；通过 MLP 投影回 d_shared
        self.aux_fusion = nn.Sequential(
            nn.Linear(d_shared + self.AUX_TOTAL, d_shared),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.aux_dropout = aux_dropout  # 训练时对辅助输入做 dropout

    def encode_aux(self, aux_attrs):
        """把 aux_attrs dict 转为 (B, AUX_TOTAL) one-hot 向量。"""
        parts = []
        for name, n_cls in self.AUX_DIMS.items():
            idx = aux_attrs[name].clone()
            idx[idx < 0] = 0  # 缺失视为类别 0（同时通过 mask 0 出来）
            parts.append(F.one_hot(idx, n_cls).float())
        return torch.cat(parts, dim=-1)

    def forward(self, batch):
        out = self.base(batch)  # 包含 participant_repr
        aux_vec = self.encode_aux(batch["aux_attrs"])
        # 训练时对辅助输入做 dropout，强迫 Teacher 不能完全依赖辅助属性
        if self.training and self.aux_dropout > 0:
            mask = (torch.rand(aux_vec.shape[0], 1, device=aux_vec.device) > self.aux_dropout).float()
            aux_vec = aux_vec * mask
        fused = self.aux_fusion(torch.cat([out["participant_repr"], aux_vec], dim=-1))
        # 重新过任务头
        out["task_logits"] = self.base.task_head(fused)
        out["participant_repr"] = fused
        return out
```

**关键设计**：辅助属性 dropout（推荐 0.2）防止 Teacher 退化为"只看辅助属性"的模型。如果 Teacher 在辅助属性被完全置零时仍然能在验证集上达到相当于 baseline 的性能，则蒸馏才有意义。

### 4.4 KD 损失（Stage B）

```python
def kd_loss(student_logits, teacher_logits, T: float = 2.5):
    """KL散度蒸馏。teacher_logits 应 detach。"""
    s = F.log_softmax(student_logits / T, dim=-1)
    t = F.softmax(teacher_logits.detach() / T, dim=-1)
    return F.kl_div(s, t, reduction="batchmean") * (T * T)
```

**A1 情形**（多标签二分类）：每个标签独立做 KD。BCE 风格的蒸馏：

```python
def kd_loss_bce(student_logits, teacher_logits, T: float = 2.5):
    s = torch.sigmoid(student_logits / T)
    t = torch.sigmoid(teacher_logits.detach() / T)
    # 等价于对每个二分类做 KL
    return F.binary_cross_entropy(s, t, reduction="mean") * (T * T)
```

**A2 情形**（CORAL / 序数 21 题项 × 3 阈值）：把 (B, 21, 3) flatten 成 (B, 63)，逐 logit 做 BCE 风格的 KD。

### 4.5 Stage B 损失合成

```python
alpha = config.aux_lupi.phase3_kd.alpha  # 推荐 0.3~0.5
T = config.aux_lupi.phase3_kd.temperature  # 推荐 2~3

with torch.no_grad():
    teacher_out = teacher_model(batch)

student_out = student_model(batch)

L_hard = main_loss(student_out["task_logits"], targets)
L_soft = kd_loss(student_out["task_logits"], teacher_out["task_logits"], T)

main_loss_combined = alpha * L_hard + (1 - alpha) * L_soft

# Phase 1 的辅助监督仍然加在 student 上（多任务正则化与蒸馏可叠加）
total_loss = main_loss_combined + aux_loss + ...
```

### 4.6 配置约定

```yaml
aux_lupi:
  phase3_kd:
    enabled: true
    teacher_checkpoint: runs/teacher_<run_id>/checkpoints/best.pt
    temperature: 2.5
    alpha: 0.4
    apply_aux_dropout_eval: false  # eval 时不做 aux dropout（拿全信号）
```

### 4.7 训练入口扩展

`train.py` 增加一个 `mode` 参数：

```bash
# Stage A：训练 Teacher
python train.py --config tasks/a1/default.yaml --mode teacher

# Stage B：训练 Student（蒸馏）
python train.py --config tasks/a1/default.yaml --mode student \
  --aux_lupi.phase3_kd.teacher_checkpoint runs/teacher_xxx/checkpoints/best.pt
```

`mode=teacher` 时构造 TeacherModel；`mode=student` 时构造常规 GroupedModel 并加载 teacher checkpoint 做蒸馏；`mode=baseline` 时禁用所有 LUPI 改动。

### 4.8 Phase 3 验证准则

1. Teacher 在验证集上的指标应**显著高于** Phase 1 模型（差距 1~3%）。如果 Teacher 不强于 Phase 1，蒸馏无意义，直接放弃 Phase 3。
2. Teacher 在 aux 全置零（推理时手动设 `aux_vec = 0`）的情况下，性能应**至少不低于** Phase 1 模型——这检验了 Teacher 没有完全依赖辅助属性。如果性能崩溃，提高 `aux_dropout` 重训。
3. Student 在验证集上应优于 Phase 1（这是 KD 的目的）。如果 Student 不如 Phase 1，可能是：温度 T 不合适（试 1.5、2、3、4）、alpha 不合适（试 0.2~0.6）、Teacher 容量过大（不应该，因为 Teacher 与 Student 同构）。
4. Student checkpoint 用 baseline `infer.py` 能正常推理。

---

## 5. Phase 4：Group DRO（可选）

### 5.1 目的

应对训练-测试分布在子群层面的偏移。通过用辅助属性定义子群并优化最差子群损失，提升模型对人群结构变化的鲁棒性。

### 5.2 子群定义

取 `aux_academic × aux_emotional` 的笛卡尔积，9 个子群。如果某子群训练样本数 < 20，合并到"其他"类。

### 5.3 损失改动（`runner.py`）

```python
def group_dro_loss(per_sample_loss, group_ids, num_groups, eta=0.5):
    """指数加权的 group loss。eta 越大越接近 max，越小越接近平均。"""
    group_losses = []
    for g in range(num_groups):
        mask = (group_ids == g)
        if mask.sum() > 0:
            group_losses.append(per_sample_loss[mask].mean())
        else:
            group_losses.append(torch.tensor(0.0, device=per_sample_loss.device))
    group_losses = torch.stack(group_losses)  # (num_groups,)

    # softmax-weighted average，等价于 EMA-based DRO
    weights = F.softmax(eta * group_losses.detach(), dim=0)
    return (weights * group_losses).sum()
```

### 5.4 配置

```yaml
aux_lupi:
  phase4_dro:
    enabled: true
    grouping_attrs: [aux_academic, aux_emotional]
    eta: 0.5
    fallback_group_id: 9  # 小样本子群合并到此
```

### 5.5 Phase 4 验证准则

1. 训练日志记录每个子群的 per-epoch loss，确认所有子群 loss 都在下降。
2. 验证集上分组报告主指标（每个子群单独的 F1/QWK）。最差子群指标应有提升（即使整体均值提升不明显）。
3. 不要为了 Phase 4 显著降低整体均值——如果整体均值下降 > 1%，回滚此阶段。

---

## 6. 验证、日志与可追溯性

### 6.1 每阶段产出

每个阶段训练完成后，`runs/<run_id>/run_meta.json` 中必须包含：

- `aux_lupi_config`：完整的 `aux_lupi` 块快照
- `aux_metrics`：辅助任务的训练 / 验证准确率（每个属性独立）
- `subgroup_metrics`：按 `aux_academic × aux_emotional` 子群划分的主指标
- `baseline_delta`：相对 baseline 的主指标变化（需要先跑一次 baseline 作为参照）

### 6.2 必须保留的 baseline 参照

在开始任何改动前，确保有一份**当前 baseline 在当前数据上的训练结果**作为对照（同 seed、同 config，仅 `aux_lupi.enabled=false`）。这是所有 Phase 验证的基准。

### 6.3 ablation 跑表

最终至少要有以下实验：

| Run | Phase 1 | Phase 2 | Phase 3 | Phase 4 | 用途 |
|-----|:-:|:-:|:-:|:-:|------|
| baseline | ❌ | ❌ | ❌ | ❌ | 基准 |
| p1 | ✅ | ❌ | ❌ | ❌ | Phase 1 净增益 |
| p1+p2 | ✅ | ✅ | ❌ | ❌ | + 样本加权 |
| teacher_p1p2 | ✅ | ✅ | (本身是 Teacher) | ❌ | Teacher checkpoint 来源 |
| student_p1p2p3 | ✅ | ✅ | ✅ | ❌ | 最终主提交候选 |
| student_p1p2p3p4 | ✅ | ✅ | ✅ | ✅ | + DRO（如果有正向收益） |

每个 run 用至少 3 个 seed（0/42/2026）取均值与标准差，避免单次划分的方差掩盖效应。

### 6.4 提交模型选择

最终提交用 `student_p1p2p3` 或 `student_p1p2p3p4` 中验证集主指标最高者；如果 ensemble 时间允许，可对 3 个 seed 的 Student 输出做概率平均。

---

## 7. 常见陷阱与避免方式

### 7.1 数据泄漏

辅助属性 CSV 必须只在 train split 加载。`val` 也不应使用辅助属性（即使有），因为验证集要模拟测试时不可见辅助属性的情况——但 Phase 1 的辅助监督在 val 上不计算损失（标签信号没用），仅 train loss 用。**Phase 3 的 Teacher 在 train 和 val 都用辅助属性**（因为 Teacher 推理时也用，它只是用于产生软标签；Student 推理时不用）。

### 7.2 辅助属性缺失值

实际数据中可能有以下缺失情形：

- 独生子女 → `aux_favoritism = -1`（结构性缺失，正常）
- 漏填 → 任一属性 = -1
- CSV 中该 PID 完全没有记录 → 5 个属性全部 -1

所有 -1 在 cross_entropy 时必须 mask 掉，**不要填 0 或众数**——会污染监督信号。

### 7.3 Teacher 退化

Teacher 训练时如果 aux dropout 太低（< 0.1），Teacher 可能学会"主要看辅助属性"。验证方法：训练后对 Teacher 做两次推理，一次正常一次把所有 `aux_attrs` 设为 -1（→ aux_vec 全零）。性能差距应 < 2%。差距过大说明 Teacher 没有充分利用音视频，蒸馏出来的 Student 会很差。

### 7.4 配置切换的副作用

每加一个 phase，先用 `enabled=false` 跑一遍确认 baseline 行为没变。再 `enabled=true` 跑。**不要同时改 enabled 和参数**——出问题难以定位。

### 7.5 推理时的 strict_load

`infer.py` 改成 `strict=False` 后，要打印 missing/unexpected keys。**unexpected** 中只应该出现 `aux_heads.*` 或 `teacher.*`；**missing** 应该为空。出现其他情况说明 checkpoint 兼容性破坏，必须修正。

### 7.6 Phase 1 辅助任务权重调优

5 个辅助任务的权重不应该平均分配。基于临床先验：

- `aux_emotional`：与 DASS 最相关，权重最高（0.20）
- `aux_academic`：抑郁前驱症状，权重次高（0.15）
- 其他三个：弱相关，权重低（0.05）

总辅助损失权重大致应为主任务的 1/3 ~ 1/2。如果观察到主任务 loss 收敛慢或最终主指标低于 baseline，**整体下调辅助权重 50%**。

### 7.7 不要触碰的部分

以下 baseline 机制**不需要修改**：

- `MTCNBackbone` 的所有内部结构（GroupAdapter, ModalityFusion, TCN, ASP）
- `ParticipantAggregator` 的三种模式
- `SessionTypeClassifier`（与新增的辅助监督共存，不冲突）
- A1Head / A2OrdinalHead / CORALHead 本身
- 校准模块（`calibration/` 下的 bias 搜索和阈值优化）
- pos_weight 计算逻辑

如果你发现要修改这些，先停下来确认是否走错方向。

---

## 8. 实施顺序建议

按以下顺序推进，每完成一项跑一次完整训练 + 验证：

```
[Day 1]   Baseline 重跑（3 seed 取均值）→ 记录基准分数
[Day 2]   Phase 1 实现 + 单元测试（前向能跑通、enabled=false 行为不变）
[Day 3]   Phase 1 训练 3 seed → 验证净增益
[Day 4]   Phase 2 实现 + 训练（在 Phase 1 基础上叠加）
[Day 5]   Teacher 训练（启用 Phase 1+2，且模型使用 TeacherModel 包装）
[Day 6]   Student 训练（KD from Teacher，启用 Phase 1+2）
[Day 7]   （可选）Phase 4 DRO 实验
[Day 8]   Ensemble + 提交准备
```

每完成一阶段，更新 `run_meta.json` 中的 `baseline_delta` 字段并提交一次实验快照。

---

## 9. 输出物清单

所有改动完成后，仓库中应新增 / 修改：

```
新增文件:
  data/aux_attributes.csv           # 辅助属性表（如果尚未提供）
  common/models/teacher_model.py    # Phase 3 Teacher 包装
  docs/AUX_LUPI_PLAN.md             # 本文档

修改文件:
  common/data/grouped_dataset.py    # Phase 1（数据层）
  common/data/grouped_collate.py    # Phase 1（collate）
  common/models/heads.py            # Phase 1（AuxAttributeHeads）
  common/models/grouped_model.py    # Phase 1（挂载辅助头）
  common/runner.py                  # Phase 1/2/3/4（损失合成）
  train.py                          # Phase 3（增加 --mode）
  infer.py                          # Phase 1（strict=False）
  tasks/a1/default.yaml             # 配置块
  tasks/a2/default.yaml             # 配置块
```

任何阶段失败或回滚的记录都应写入 `docs/EXPERIMENTS_LOG.md`，包括失败的配置、失败的指标、回滚的原因。这是论文方法章节的素材来源。
