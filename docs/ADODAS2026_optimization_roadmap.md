# ADODAS2026 优化路线指导文档（面向 Agent 执行版）

> **文档定位**：基于现有 baseline（MTCN + ASP + CORAL/A1Head + 校准搜索）的渐进式优化路线，分三阶段推进。
> **主线**：A2 Mean QWK 提升；**副线**：A1 Macro-F1 顺带优化。
> **执行原则**：所有新增模块必须可配置切换、可独立 A/B、可回滚。具体代码实现由 agent 根据项目实际结构决定，本文档只规约"做什么、为什么、验收什么"。

---

## 0. 基线现状与核心瓶颈（必读）

### 0.1 当前结构关键点
- **任务头**：A1 → `A1Head`（BCE+pos_weight+bias 网格校准）；A2 → `CORALHead`（默认）/ `A2OrdinalHead`（可选），三种解码 + 阈值 offset 校准。
- **主干**：`MTCNBackbone` = GroupAdapter → ModalityFusion → 模态独立 TCN（6 层，dilated）→ ASP（带 VAD/QC 偏置）→ Fusion MLP。
- **聚合**：`ParticipantAggregator`（默认 mlp）+ `SessionTypeClassifier` 辅助任务。
- **训练**：AdamW + warmup-cosine + AMP + label smoothing + feature noise + session dropout（A2 已启用，A1 全关）。
- **校准**：A1 bias 网格；A2 解码策略 + 阈值 offset 联合搜索（按 QWK 排序）。

### 0.2 已识别的关键瓶颈（按修复优先级）

| # | 瓶颈 | 修复阶段 |
|---|------|----------|
| B1 | **MTL 辅助任务信息冗余**：valence/arousal 和 emotion_cls 由 DASS-21 主标签 deterministic 推导（见 `grouped_dataset.py:292-347`），等价于把主标签换形式喂网络，无新增信号；AU head 标签恒为零（占位实现），属于死分支 | 阶段一 |
| B2 | A2 损失只用 CORAL+BCE 阈值，未利用类别不平衡 / 序数距离信息（21 题 × 4 级长尾严重） | 阶段一 |
| B3 | A1 损失仅 BCE+pos_weight，对极端不平衡的少数阳性类利用不充分 | 阶段一 |
| B4 | A2 推理端无概率先验注入，0/3 极端类容易被高估 | 阶段一 |
| B5 | ASP 池化音视频独立做，跨模态信息在池化阶段尚未交互 | 阶段一 |
| B6 | TCN 仅捕获等间隔局部依赖，缺长程全局建模 | 阶段二 |
| B7 | 现融合是 concat-then-MLP 的浅融合，模态间交互弱 | 阶段二 |
| B8 | 序数回归损失仍是代理目标（BCE 阈值），与评测指标 QWK 间存在 gap | 阶段二 |
| B9 | 主干容量仅 2–5M，对小样本下的复杂模式表达力受限 | 阶段三 |
| B10 | 多任务梯度冲突（启用 MTL 后）未显式治理 | 阶段三 |

### 0.3 跨阶段不变量（每个 PR 都必须保持）
- **可切换性**：所有新增模块通过 `tasks/*/default.yaml` flag 启停，默认值保留 baseline 行为。
- **可回溯**：`run_meta.json` 中记录损失类型、MTL 权重策略、池化/时序/融合架构、解码策略。
- **校准链路兼容**：A1 bias 校准、A2 解码 + offset 搜索必须仍可运行（阶段一新增的概率先验偏置在校准之前应用）。
- **缺失模态鲁棒**：所有新模块在 session_valid=False 的 dummy slot 下要 graceful fallback（不能 NaN）。
- **小样本评估纪律**：每个关键实验**至少 3 seed 平均**报告 QWK / MAE / Macro-F1；附 per-item QWK 方差。

---

## 1. 阶段一：低风险高确定性优化

> **目标**：在不动主干骨架的前提下，通过损失体系、MTL 修复、推理端校正、池化升级，将 A2 val QWK 较 baseline 提升 ≥ 0.01；A1 Macro-F1 ≥ baseline + 0.005。

### 1.1 损失函数体系重构（B2 / B3）

#### 1.1.1 落地策略（推荐版）

将损失实现为**插件式可切换模块**，配置驱动；同时给出推荐起点，避免组合爆炸。

**推荐执行顺序**：

| 步骤 | 任务 | 变更 | 对比基线 |
|------|------|------|----------|
| Step1 | 基线复现 | 不变 | — |
| Step2 | A2 | `CORAL` → `CORN` + Class-Balanced 加权（β=0.999 起调） | vs Step1 |
| Step3 | A1 | BCE+pos_weight → **ASL**（Asymmetric Loss） | vs Step1 |
| Step4 | A1+A2 | 二者组合 | vs Step2、Step3 |
| Step5（可选） | A2 | 在 Step2 基础上叠加 **LDAM** margin 项做长尾补强 | vs Step2 |

**为什么选 CORN+CB 作为 A2 起点**：
- **CORN** 相较 CORAL **不共享 score 投影**，对 21 个 item 内部异构（不同题目难度不同）友好，且天然保留单调性的条件概率链；
- **Class-Balanced** 按 effective sample number 加权，更适合 21 × 4 长尾分布（按 item × bin 联合统计 effective N）。

**为什么 A1 选 ASL**：
- ASL 是为多标签不平衡量身设计，对负样本主导的"硬负例"做不对称 focal-style 抑制，比 BCE+pos_weight+硬截断 [1,4] 上限更柔和。

**候选保留（agent 必须实现为可选项）**：
- Focal Loss（含可调 γ）
- LDAM Loss
- Class-Balanced（β 可调）
- 原 CORAL / 原 BCE（作为回退）

#### 1.1.2 验收
- 每个 step 单独 ablation，3-seed 平均；
- 任一替换损失若使 QWK 退化 > 0.005 → **不进入下一步**，回滚至上一最优组合；
- 最终保留组合写入 `run_meta`。

---

### 1.2 MTL 重构：先修标签，再加权（B1，最关键）

#### 1.2.1 必须先处理的现状问题
当前 valence/arousal、emotion_cls 是主标签的 deterministic 再参数化，AU 是死分支。**直接在此基础上加 Uncertainty Weighting 等于把网络注意力分散到没有新信息的副本任务上，会损害主任务**。

#### 1.2.2 改造策略（强制按以下顺序）

**Stage 1-MTL-A：清理冗余（必做）**

| 任务 | 处理 |
|------|------|
| `emotion_cls`（DASS 阈值推导） | **删除** —— 与主任务 100% 同源 |
| `valence/arousal`（DASS 线性推导） | **降级为正则化损失**，权重压到 ≤ 0.05；或同样删除 |
| `AU 预测`（占位零） | **暂时禁用 head**（不再训练空标签） |

**Stage 1-MTL-B：引入真实增量信号（推荐做，但允许延后）**

| 任务 | 数据来源（伪标签） | 接入方式 |
|------|---------------------|----------|
| 面部 AU | OpenFace 2.x 或 Py-Feat 离线预提取，存为 parquet（与现有 `face_meta` 同级） | 作为 12d AU 强度向量的回归监督 |
| 语音情感分类（SER） | 外部预训练 SER 模型（如 `emotion2vec` / `funASR-emotion`）打帧级 / 段级软标签 | 4-7 类离散 + 软标签分类 |
| 维度情感（VA） | 外部 VA 回归模型（如 wav2vec2-MSP-Podcast、ABAW 系预训练）打帧级 valence/arousal | 替换当前 deterministic 推导的 VA |

> agent 在评估资源后选择实际接入哪几个；至少接入 1 个真实辅助任务，否则 MTL 部分不计入阶段一交付。

**Stage 1-MTL-C：启用 Uncertainty Weighting**

仅在 Stage 1-MTL-A 完成（且 1-MTL-B 至少一个任务接入）后启用 Kendall 2018 风格的可学习 σ²：

- 每个任务一个可学习 log_var 参数；
- 总损失：`Σ_i [exp(-s_i) * L_i + s_i]`，其中 `s_i = log σ_i²`；
- 主任务（A1 或 A2）和强相关辅助任务（VA、AU、SER）权重均由 σ² 自适应；
- 现有 `session_loss` 与 `session_type_loss` 是否纳入同一框架由 agent 评估（建议纳入但赋更小学习率）。

#### 1.2.3 验收
- MTL 改造前后必须分别报告主任务指标，**MTL 必须带来 ≥ 0.005 的 QWK 增益**才保留；
- 若仅做 Stage 1-MTL-A（删冗余）就已经使 QWK 提升，说明原 MTL 在拖后腿，这本身是有价值的负面发现，需记录。

---

### 1.3 A2 推理端：类别先验偏置（B4）

#### 1.3.1 设计
在 A2 推理时（**不影响训练**），对每题项预测概率（或 CORAL 累积概率）做先验校正：

- 先用训练集（或 train+val）统计每题项的类别先验 `prior_k`（k ∈ {0,1,2,3}）；
- 在推理前对类别 k 的概率乘以 `weight_k`，其中：
  - 中间类（k=1, 2）乘 `α_mid`（>1）
  - 极端类（k=0, 3）乘 `α_ext`（<1）
- `α_mid, α_ext` 在验证集上做小范围网格搜索（如 1.0~1.5 / 0.7~1.0，步长 0.05），主排序 QWK，次排序 MAE。

#### 1.3.2 与现有校准链路的关系
- **执行顺序**：模型 logits → softmax/累积概率 → **先验偏置（新增）** → 现有解码（argmax/monotonic/expectation）→ 现有 offset 校准；
- 偏置参数写入与现有 `a2_threshold_offsets_grouped.json` 同级或合并到同一 JSON；
- `infer.py` 读取并复现。

#### 1.3.3 验收
- 混淆矩阵：0↔3 误判频次下降、1↔2 召回上升；
- QWK 单独提升 ≥ 0.005 才保留；
- 若 MAE 上升（典型 trade-off），允许接受小幅 MAE 退化（< 0.02），但需在 run_meta 明确记录。

---

### 1.4 池化层升级：ASP → Cross-Modal ASP（B5）

#### 1.4.1 设计要点
现 ASP 的 attention score 仅由本模态序列 + VAD/QC 偏置生成；改造为：

- 音频 ASP 的注意力打分 query 由 **video 全局向量** 参与（如 video TCN 输出的 mean pooling 或一个 [CLS] 向量）；
- 视频 ASP 反向同理；
- 保留原 VAD/QC 偏置项不变（不要破坏现有显式先验注入）；
- 跨模态 query 通过一个轻量 linear gate 控制（避免某一模态完全缺失时打挂另一模态）。

#### 1.4.2 鲁棒性约束
- 当某 session 的某模态完全无效（全 mask=0）时，cross-modal query 应**退化为本模态自打分**（即等价回原 ASP）；
- 建议加 dropout 在 cross-modal gate 上，避免模型过度依赖单边。

#### 1.4.3 验收
- 消融对比 ASP vs CM-ASP（其他不变）；
- A2 val QWK Δ ≥ 0.005 才保留，否则继续用 ASP。

---

### 1.5 阶段一汇总验收

| 子项 | 单独验收门槛 | 关键产出 |
|------|---------------|----------|
| 损失体系 | A2 QWK +0.005 或 A1 F1 +0.005 | 可切换的 Loss 注册表 |
| MTL 修复 | A2 QWK +0.005（前提：MTL-A 已完成） | 删除冗余/接入真实辅助/UW 配置 |
| 概率先验偏置 | A2 QWK +0.005 | 推理校准 JSON 扩展字段 |
| CM-ASP | A2 QWK +0.005 | 配置 flag `pooling: cm_asp` |
| **整体合并** | **A2 val QWK ≥ baseline + 0.01** | 综合配置 + 复现脚本 |

阶段一完成前**不进入阶段二**。

---

## 2. 阶段二：中等风险中等增益优化

> **目标**：在阶段一最佳模型上，将 A2 val QWK 再提升 ≥ 0.015；并保持或提升 A1 Macro-F1。

### 2.1 时序建模升级：TCN → 混合时序编码器（B6）

#### 2.1.1 候选方案与推荐
| 方案 | 形态 | 优点 | 风险 |
|------|------|------|------|
| **A（推荐起点）** | TCN（3-4 层）+ Transformer（2-3 层）串联 | 局部+全局互补，改动可控 | 序列较长时 attention 显存压力 |
| B | TCN + LSTM/GRU 并联 | 训练稳定 | 长序列循环退化 |
| C | MS-S-TCN（多尺度共享 TCN） | 改动最小，无新参数族 | 增益上限低 |
| D | 纯 FlashAttention Transformer + 相对位置编码 | 长程建模强 | 小样本易过拟合 |

**推荐顺序**：先做 A 与 C 的 ablation，A 不显著优于 C 时优先保留 C（更安全），否则上 A。

**音视频跨模态时序对齐**：在 A 方案的 Transformer 部分可加 **Cross-Attention 分支**（音频序列与视频序列互为 K/V），与 1.4 的 CM-ASP 形成时序级+池化级双层跨模态对齐。

#### 2.1.2 工程约束
- 沿用现 `valid_mask`，attention mask 必须正确广播到 padding 位置；
- 序列长度截断/分块策略需在 dataset 层或 collate 层明确（避免单 session 过长爆显存）；
- 与现有 `feature_noise`、`session_drop` 仍兼容。

#### 2.1.3 验收
- 单独消融：A2 QWK Δ ≥ 0.008；
- 训练耗时增量 < 2× baseline；否则改用方案 C。

---

### 2.2 跨模态融合升级：MulT（B7）

#### 2.2.1 设计要点
- 用 MulT（Multimodal Transformer）替换/增强现 Fusion MLP；
- 至少实现 A→V、V→A 两个 cross-modal attention 分支；
- 与阶段一 CM-ASP 形成 "深-浅互补"：CM-ASP 在池化前做轻量 gating，MulT 在表示层做深层互注；
- 输入：模态独立时序编码器输出（来自 2.1）；
- 输出：与现 `session_repr` 接口一致（不破坏下游 `ParticipantAggregator`）。

#### 2.2.2 与现 ASP/CM-ASP 的执行序
建议两种结构同时实验：
- **结构A**：时序编码 → MulT 跨模态 → CM-ASP 池化；
- **结构B**：时序编码 → CM-ASP 池化 → MulT 处理池化后表示（更轻）；

二选一保留。

#### 2.2.3 验收
- 在 2.1 最佳时序基础上单独消融 MulT；
- QWK Δ ≥ 0.005 才保留。

---

### 2.3 序数回归进一步优化（B8）

#### 2.3.1 可微 QWK 辅助损失
- 实现 differentiable QWK（soft-QWK，基于 expected confusion matrix）；
- 作为**辅助损失**与 CORN 主损失加权联合（建议 `loss = L_corn + λ_qwk * L_softqwk`，λ 从 0.1 起调）；
- **必须 warmup**：前若干 epoch 仅用 L_corn，待主任务稳定后再注入 soft-QWK，否则数值不稳；
- 建议监控梯度范数，必要时 clip。

#### 2.3.2 单峰分布正则（Unimodal Regularization）
- 约束输出概率分布关于真值 y_i 单峰（即 p(k) 随 |k-y_i| 单调下降）；
- 实现可选：
  - 二项分布参数化：用一个 logit 直接生成 binomial 形式的概率分布；
  - 软单峰惩罚：对违反单峰的相邻 bin 概率差加 hinge 惩罚；
- 与 CORN 的单调性约束**互补**（CORN 约束的是 P(y≥k)，单峰约束的是 P(y=k)）；
- 仅在阶段一 + 2.1 + 2.2 都完成后再叠加，单独消融。

#### 2.3.3 验收
- soft-QWK 单独验收：QWK +0.003 起步；
- 单峰正则单独验收：QWK +0.003 起步且 MAE 不退化；
- 不达标的子项不保留。

---

### 2.4 阶段二汇总验收

| 子项 | 单独门槛 | 累积门槛 |
|------|----------|----------|
| 混合时序 | QWK +0.008 | — |
| MulT 融合 | QWK +0.005 | — |
| soft-QWK / unimodal | QWK +0.003 | — |
| **阶段二总验收** | — | **A2 val QWK ≥ 阶段一最佳 + 0.015** |

阶段二完成前**不进入阶段三**。

---

## 3. 阶段三：高风险高上限优化

> **目标**：探索 SOTA 上限，最终模型可能由 1-2 个阶段三子项组成 ensemble；不要求每个子项都进入最终提交。

### 3.1 SOTA Backbone：Conformer / Mamba-VA（B9）

#### 3.1.1 Conformer（音频侧首选）
- 用 Conformer block 替换音频侧时序编码器（或与 2.1 的混合时序串联）；
- Conformer = MHSA + Conv 模块，对韵律/局部声学模式天然友好；
- 注意参数规模：Conformer-small（约 10–20M）已可能超过现 backbone 总和，需要：
  - 重新调 LR / weight decay；
  - 启用更强正则（增大 feature_noise、引入 SpecAugment-style 频域增强）；
  - 重新评估早停策略。

#### 3.1.2 Mamba-VA / SSM（视频或聚合层备选）
- 状态空间模型对超长序列推理友好；
- 建议先在视频侧实验（视频序列通常更长且更稀疏）；
- 与 Conformer 互补：Conformer 在音频，Mamba 在视频。

#### 3.1.3 验收
- 与阶段二最佳模型对比；
- QWK +0.01 才保留；否则只作为 ensemble 候选。

---

### 3.2 MPCF 融合架构

- Multimodal Progressive Co-Fusion：在 MulT 之上引入逐层 modality-specific + modality-shared 通道；
- 替换或叠加在 MulT 上（agent 评估结构合理性后决定）；
- 与 Conformer/Mamba 的输出对接；
- 验收：QWK +0.01 才保留。

---

### 3.3 梯度冲突治理：DB-MTL + PCGrad（B10）

#### 3.3.1 启用条件
- 仅在阶段一 MTL 已完成且确实启用了 ≥ 2 个真实辅助任务时考虑；
- 与 Uncertainty Weighting 是不同层面：UW 调权重，PCGrad 调方向，可叠加。

#### 3.3.2 设计要点
- **PCGrad**：每个任务独立 backward 得到梯度，对冲突梯度做投影；
- **DB-MTL**：动态平衡任务梯度幅度；
- **代价**：每 step 多次 backward，吞吐显著下降（约 2-3×）；
- 实现时注意 AMP/梯度缩放器与多任务 backward 的兼容。

#### 3.3.3 验收
- 不要求 QWK 单独提升 ≥ 0.005，但要求**多任务整体指标稳定性提升**（多 seed 方差下降）；
- 若启用后训练耗时不可接受，可仅在最后阶段精调时启用。

---

### 3.4 阶段三汇总验收

阶段三不设硬性总门槛；交付物为：
- 完整对比矩阵（各 backbone × 融合 × 损失组合）；
- 最终 ensemble 候选清单（建议 3-5 个模型多样性组合）；
- 在 test_hidden 提交对应的最优单模型 + ensemble 两套结果。

---

## 4. 全局工程纪律

### 4.1 配置与可切换性
- 每个阶段新增模块必须有对应 yaml flag，默认值保留 baseline 行为；
- yaml flag 命名建议保持层次化（如 `loss.a2.type: corn`、`model.pooling: cm_asp`、`model.temporal.encoder: tcn_transformer`）；
- 同一 run_id 下不允许中途切换关键 flag。

### 4.2 实验追踪
- `run_meta.json` 扩展字段：损失类型、MTL 任务列表+权重策略、池化类型、时序架构、融合架构、推理校准链路；
- 命名规范建议：`{track}_{loss}_{pool}_{temporal}_{fusion}_{mtl}_seed{n}`，例：`a2_corn_cmasp_tcntfm_mult_uw_seed42`。

### 4.3 评估纪律
- **必做**：每个关键实验 ≥ 3 seed，报告 mean ± std；
- 主指标：A2 mean QWK / A1 macro F1；
- 辅助指标：A2 mean MAE、per-item QWK 方差、混淆矩阵；
- 若某子项 3 seed 间 std 大于 mean 增益的 50%，视为**不显著**，不保留。

### 4.4 校准链路兼容性（强约束）
- 现有 A1 bias 网格、A2 解码 + offset 搜索必须仍可运行；
- 阶段一新增的"概率先验偏置"在解码之前应用；
- 校准产物文件路径与字段保持向后兼容（新字段以可选形式追加）。

### 4.5 缺失模态/缺失 session 鲁棒性（强约束）
- 所有新增模块（CM-ASP、MulT、Cross-Attention、MPCF）在缺失模态时必须 graceful fallback，**不能产生 NaN/Inf**；
- 单元测试覆盖：单模态 session、单 session participant、全模态缺失边界。

### 4.6 复现性
- 每个阶段末尾打包：最佳 checkpoint + config_used.yaml + 校准 JSON + run_meta.json；
- `infer.py` 路径不变，可直接加载阶段一/二/三任一交付物。

---

## 5. 风险清单与回滚预案

| 风险 | 阶段 | 信号 | 回滚预案 |
|------|------|------|----------|
| MTL 引入真实辅助任务后仍无收益 | 一 | QWK Δ < 0.003 | 仅保留 Stage 1-MTL-A（删冗余），不启用 UW |
| CM-ASP 在缺模态 session 上 NaN | 一 | val loss 飙升 | 自打分 fallback 强制启用 |
| 概率先验偏置使 MAE 显著上升 | 一 | MAE +0.05 以上 | 仅保留中间值 boost，去掉极端值 dampen |
| 混合时序在小样本上过拟合 | 二 | train-val gap 拉大 | 退回方案 C（MS-S-TCN）或加大正则 |
| 可微 QWK 训练发散 | 二 | grad norm 爆炸 | 延长 warmup、降 λ 或暂时禁用 |
| Conformer 训练超出显存/时间预算 | 三 | OOM / 单 epoch > 2 小时 | 退回 Conformer-tiny 或仅在音频侧用 |
| PCGrad 训练耗时不可接受 | 三 | step 时间 > 3× | 仅最后 finetune 阶段启用 |

---

## 6. 待用户后续决策事项

> 这些不阻塞阶段一启动，但执行到对应子项前需明确：

1. **AU/SER/VA 真实伪标签** 的具体外部模型选型（agent 可先给候选清单再由用户决策）；
2. 阶段三 **GPU 预算上限**（影响 Conformer/MulT 规模与 PCGrad 是否启用）；
3. 是否允许在阶段二/三引入 **音频/视频 SSL 模型的微调**（当前 baseline 全部冻结）；
4. 最终提交是单模型还是 ensemble，若 ensemble 是否限制模型数。

---

## 7. 阶段间依赖图

```
阶段一（必须全部通过验收）
├── 损失体系重构  ─┐
├── MTL 修复 ────┤
├── 概率先验偏置 ─┼─► 阶段一最佳模型 ──► 阶段二起点
└── CM-ASP ─────┘

阶段二（基于阶段一最佳）
├── 混合时序 ────┐
├── MulT 融合 ───┼─► 阶段二最佳模型 ──► 阶段三起点
└── soft-QWK/单峰 ┘

阶段三（探索性，允许部分子项不进入最终模型）
├── Conformer/Mamba ─┐
├── MPCF ───────────┼─► 最终单模型 + ensemble 候选
└── DB-MTL/PCGrad ──┘
```

---

---

## 附录 A：实施状态 (2026-05-26 审计)

### A.1 已完成项

| 路线图子项 | 实现位置 | 说明 |
|-----------|---------|------|
| ASL 损失 (A1) | `common/models/heads.py:176` | `asymmetric_loss()` — 生产化 |
| CORN 损失 (A2) | `common/models/heads.py:308` | `corn_loss()` — A2 default.yaml 默认启用 |
| 可微 QWK 损失 (A2) | `common/models/heads.py:357` | `differentiable_qwk_loss()` — A2 default.yaml 默认启用 |
| Uncertainty Weighting | `common/models/mtl_uncertainty.py:28` | Kendall 2018 log_var — 默认关闭 |
| A2 阈值 offset 校准 | `common/runner.py:1081-1117` | 逐题 grid search, 推理时 logits+offsets |
| A1 bias 网格校准 | `common/runner.py:1120-1137` | 生产化 |
| ASL+Soft-F1 联合 (A1) | `common/models/heads.py:262` | `a1_loss()` |
| CORN+QWK 联合 (A2) | `common/models/heads.py:424` | `a2_ordinal_loss()` — 默认启用 |
| Phase1 MTL wrapper | `common/models/phase1_integration.py` | OptimizedGroupedModel |

### A.2 待办 (按优先级)

| # | 子项 | 路线图引用 | 状态 |
|---|------|-----------|------|
| P0 | MTL 冗余清理 (删 emotion_cls/au_pred, 降级 emotion_dims) | B1 / 1.2 | 🟢 已完成 (2026-05-26) |
| P1 | Cross-Modal ASP 池化 | B5 / 1.4 | 🔴 待执行 |
| P2 | A2 推理端类别先验偏置 | B4 / 1.3 | 🔴 待执行 |
| P3 | LDAM / Class-Balanced 损失 | B2 / 1.1 Step5 | 🔴 待执行 |
| P4 | 混合时序编码器 (TCN+Transformer) | B6 / 2.1 | 🔴 待执行 |
| P5 | MulT 跨模态融合 | B7 / 2.2 | 🔴 待执行 |
| P6 | 单峰分布正则 | B8 / 2.3.2 | 🔴 待执行 |
| P7 | 真实辅助任务伪标签 (AU/SER/VA) | 1.2 Stage 1-MTL-B | 🔴 待执行 |
| P8 | 阶段三探索 (Conformer/Mamba/PCGrad) | 3.1-3.3 | 🔴 待执行 |

### A.3 不做的

- **Focal Loss 独立实现**：ASL 已是 Focal Loss 超集
- **HDF5 辅助标签支持**：当前 MTL 在 HDF5 下 aux_targets=None 可正常运行；待 P7 有外部标签后一并添加

---

**文档版本**：v1.1 (追加实施状态审计)
**适用范围**：ADODAS 2026 Challenge baseline 优化
**变更纪律**：本文档为执行规约，agent 在执行中若发现某子项无法按预期实施，需先回写文档（附阻塞原因）再决定方案变更。
