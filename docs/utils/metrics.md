# metrics.py - 评估指标计算详解

## 文件概述

`metrics.py` 实现了 A1 和 A2 任务所需的评估指标计算函数，包括 F1-score、AUROC、QWK 和 MAE 等。

## A1任务指标

### binary_f1() - 二元F1分数

```python
def binary_f1(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> float:
    """
    计算多标签二元分类的平均F1分数
    
    参数:
        probs: (N, C) 预测概率矩阵
        labels: (N, C) 真实标签矩阵 (0或1)
        threshold: 分类阈值
    
    返回:
        所有类别的平均F1分数
    """
    preds = (probs >= threshold).astype(int)
    scores = []
    for c in range(probs.shape[1]):
        scores.append(f1_score(labels[:, c], preds[:, c], zero_division=0.0))
    return float(np.mean(scores))
```

**F1分数计算原理**：

$$F1 = \frac{2 \times Precision \times Recall}{Precision + Recall}$$

其中：
- Precision = TP / (TP + FP) - 预测为正的样本中有多少是真正
- Recall = TP / (TP + FN) - 真正为正的样本有多少被正确预测

**多标签场景**：
- A1任务有3个标签 (D, A, S)
- 计算每个标签的F1，然后取平均
- `zero_division=0.0` 处理无正样本的情况

### per_class_f1() - 每类F1分数

```python
def per_class_f1(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> list[float]:
    """
    返回每个类别的F1分数列表
    
    返回: [F1_D, F1_A, F1_S]
    """
    preds = (probs >= threshold).astype(int)
    return [
        float(f1_score(labels[:, c], preds[:, c], zero_division=0.0))
        for c in range(probs.shape[1])
    ]
```

**用途**：分析模型在每个类别上的表现，识别薄弱环节

### macro_auroc() - 宏平均AUROC

```python
def macro_auroc(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    计算多标签的宏平均AUROC
    
    AUROC (Area Under ROC Curve): ROC曲线下面积
    衡量模型区分正负样本的能力
    """
    scores = []
    for c in range(probs.shape[1]):
        unique = np.unique(labels[:, c])
        if len(unique) < 2:
            # 如果只有一个类别，AUROC无定义
            scores.append(0.0)
        else:
            scores.append(float(roc_auc_score(labels[:, c], probs[:, c])))
    return float(np.mean(scores))
```

**AUROC 解释**：

```
ROC曲线:
- X轴: False Positive Rate (FPR)
- Y轴: True Positive Rate (TPR)
- 通过改变分类阈值绘制

AUROC含义:
- 1.0: 完美分类器
- 0.5: 随机猜测
- < 0.5: 比随机还差
```

**示例**：
```python
probs = [[0.9, 0.3, 0.8],   # 样本1: 预测D=0.9, A=0.3, S=0.8
         [0.2, 0.7, 0.4]]   # 样本2
labels = [[1, 0, 1],        # 样本1: D=1, A=0, S=1
          [0, 1, 0]]        # 样本2

mf1 = binary_f1(probs, labels, threshold=0.5)  # 平均F1
pcf1 = per_class_f1(probs, labels)  # [F1_D, F1_A, F1_S]
auroc = macro_auroc(probs, labels)  # 宏平均AUROC
```

## A2任务指标

### _quadratic_weighted_kappa() - 二次加权Kappa

```python
def _quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 4) -> float:
    """
    计算二次加权Kappa系数
    
    QWK考虑预测值与真实值之间的距离
    距离越远，惩罚越大
    """
    N = num_classes  # 4个类别: 0, 1, 2, 3
    
    # 1. 构建权重矩阵
    # w[i,j] = (i-j)² / (N-1)²
    w = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            w[i, j] = (i - j) ** 2 / ((N - 1) ** 2)
```

**权重矩阵可视化**：

```
权重矩阵 w (N=4):
          预测值
         0    1    2    3
    0 [ 0.00 0.11 0.44 1.00 ]
真  1 [ 0.11 0.00 0.11 0.44 ]
实  2 [ 0.44 0.11 0.00 0.11 ]
值  3 [ 1.00 0.44 0.11 0.00 ]

对角线权重=0 (预测正确)
距离越远权重越大
```

```python
    # 2. 构建混淆矩阵 O (观察到的)
    O = np.zeros((N, N), dtype=np.float64)
    for t, p in zip(y_true, y_pred):
        O[t, p] += 1
    
    # 3. 构建期望矩阵 E (随机情况)
    hist_true = np.bincount(y_true, minlength=N)  # 真实值分布
    hist_pred = np.bincount(y_pred, minlength=N)  # 预测值分布
    E = np.outer(hist_true, hist_pred) / len(y_true)
    
    # 4. 计算QWK
    # QWK = 1 - (Σw×O) / (Σw×E)
    num = np.sum(w * O)
    den = np.sum(w * E)
    if den == 0:
        return 1.0  # 完美预测
    return 1.0 - num / den
```

**QWK 计算示例**：

```
真实值: [0, 1, 2, 3, 2, 1]
预测值: [0, 1, 3, 3, 2, 0]

混淆矩阵 O:
          预测
         0   1   2   3
    0 [  1   0   0   0  ]
真  1 [  1   1   0   0  ]
实  2 [  0   0   1   1  ]
    3 [  0   0   0   1  ]

解释:
- 真实=0, 预测=0: 1次 (正确)
- 真实=1, 预测=0: 1次 (错误，距离=1)
- 真实=1, 预测=1: 1次 (正确)
- 真实=2, 预测=2: 1次 (正确)
- 真实=2, 预测=3: 1次 (错误，距离=1)
- 真实=3, 预测=3: 1次 (正确)

加权错误 = w[1,0] + w[2,3] = 0.11 + 0.11 = 0.22
QWK ≈ 1 - 某个小于1的值 = 接近1
```

**QWK 值解读**：
- 1.0: 完美预测
- 0.8-1.0: 高度一致
- 0.6-0.8: 中等一致
- 0.4-0.6: 一般一致
- < 0.4: 较差
- < 0: 预测比随机还差

### mean_qwk() - 平均QWK

```python
def mean_qwk(preds: np.ndarray, labels: np.ndarray) -> float:
    """
    计算所有项目的平均QWK
    
    参数:
        preds: (N, 21) 预测值矩阵
        labels: (N, 21) 真实值矩阵
    
    返回:
        21个项目的平均QWK
    """
    scores = []
    for c in range(preds.shape[1]):
        scores.append(_quadratic_weighted_kappa(labels[:, c], preds[:, c]))
    return float(np.mean(scores))
```

### per_item_qwk() - 每项QWK

```python
def per_item_qwk(preds: np.ndarray, labels: np.ndarray) -> list[float]:
    """
    返回每个项目的QWK分数
    
    返回: [QWK_d01, QWK_d02, ..., QWK_d21]
    """
    return [
        _quadratic_weighted_kappa(labels[:, c], preds[:, c])
        for c in range(preds.shape[1])
    ]
```

**用途**：识别哪些心理评估项目预测较难

### mean_mae() - 平均MAE

```python
def mean_mae(preds: np.ndarray, labels: np.ndarray) -> float:
    """
    计算所有项目的平均MAE
    
    MAE = Σ|y_true - y_pred| / N
    """
    scores = []
    for c in range(preds.shape[1]):
        scores.append(float(mean_absolute_error(labels[:, c], preds[:, c])))
    return float(np.mean(scores))
```

**MAE 特点**：
- 直观：平均预测偏差
- 对异常值敏感
- 与QWK互补

## 指标对比分析

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           指标对比分析表                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  指标       │  值范围  │  适用任务  │  优点                │  缺点            │
│  ─────────────────────────────────────────────────────────────────────────  │
│  F1        │  0-1    │  A1       │  平衡精确率和召回率   │  不考虑阈值变化 │
│  AUROC     │  0-1    │  A1       │  阈值无关            │  不直接反映分类  │
│  QWK       │  -1~1   │  A2       │  考虑序数距离        │  可能掩盖单类错误│
│  MAE       │  0-∞    │  A2       │  直观误差度量        │  对异常敏感     │
│                                                                             │
│  选择建议:                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  A1任务: 主要看 F1 (提交排名), AUROC (模型比较)                        │  │
│  │  A2任务: 主要看 QWK (提交排名), MAE (误差分析)                         │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 校准相关指标

### A1校准

```python
def calibrate_a1_bias(logits, labels, grid_min=-3.0, grid_max=3.0, grid_step=0.1):
    """
    通过网格搜索找到最佳偏置值，优化F1分数
    
    原理:
    - 模型输出的logit可能存在偏移
    - 通过添加偏置调整预测概率
    - 在验证集上搜索最优偏置
    """
    grid = np.arange(grid_min, grid_max + grid_step, grid_step)
    biases = np.zeros(3)
    best_f1s = []
    
    for t in range(3):  # 对每个类别独立优化
        best_f1 = -1.0
        best_b = 0.0
        for b in grid:
            probs = 1.0 / (1.0 + np.exp(-(logits[:, t] + b)))
            preds = (probs > 0.5).astype(int)
            f1 = f1_score(labels[:, t], preds, zero_division=0.0)
            if f1 > best_f1:
                best_f1 = f1
                best_b = b
        biases[t] = best_b
        best_f1s.append(best_f1)
    
    return biases, best_f1s
```

**校准效果**：
```
校准前:
  偏置 = [0, 0, 0]
  F1 = 0.65

校准后:
  偏置 = [-0.5, +0.3, -0.2]  (找到最优偏置)
  F1 = 0.72  (提升)
```

### A2校准

```python
def calibrate_a2_thresholds(logits, labels, n_items=21, n_thresholds=3,
                             grid_min=-2.0, grid_max=2.0, grid_step=0.1,
                             decode_method="expectation"):
    """
    通过网格搜索找到最佳阈值偏移，优化QWK
    
    原理:
    - 序数回归的阈值可能有偏移
    - 添加统一偏移调整预测
    """
    grid = np.arange(grid_min, grid_max + grid_step, grid_step)
    offsets = np.zeros((n_items, n_thresholds))
    
    for j in range(n_items):
        best_qwk = -1.0
        best_offset = np.zeros(n_thresholds)
        
        # 对每个项目搜索最佳偏移
        for b in grid:
            shifted = logits[:, j, :] + b
            preds = decode_function(shifted)
            qwk = cohen_kappa_score(labels[:, j], preds, weights="quadratic")
            if qwk > best_qwk:
                best_qwk = qwk
                best_offset = np.full(n_thresholds, b)
        
        offsets[j] = best_offset
    
    return offsets, item_qwks
```

## 使用示例

### A1评估

```python
from common.utils.metrics import binary_f1, per_class_f1, macro_auroc

# 收集验证集预测
all_probs = []
all_labels = []
for batch in val_loader:
    with torch.no_grad():
        logits = model(batch)
        probs = torch.sigmoid(logits).cpu().numpy()
    all_probs.append(probs)
    all_labels.append(batch["y_a1"].numpy())

probs = np.concatenate(all_probs)
labels = np.concatenate(all_labels)

# 计算指标
mf1 = binary_f1(probs, labels, threshold=0.5)
pcf1 = per_class_f1(probs, labels)
auroc = macro_auroc(probs, labels)

print(f"Mean F1: {mf1:.4f}")
print(f"Per-class F1: D={pcf1[0]:.4f}, A={pcf1[1]:.4f}, S={pcf1[2]:.4f}")
print(f"Macro AUROC: {auroc:.4f}")
```

### A2评估

```python
from common.utils.metrics import mean_qwk, per_item_qwk, mean_mae

# 收集验证集预测
all_preds = []
all_labels = []
for batch in val_loader:
    with torch.no_grad():
        logits = model(batch)
        preds = A2OrdinalHead.predict_expectation(logits).cpu().numpy()
    all_preds.append(preds)
    all_labels.append(batch["y_a2"].numpy())

preds = np.concatenate(all_preds)
labels = np.concatenate(all_labels)

# 计算指标
mqwk = mean_qwk(preds, labels)
piqwk = per_item_qwk(preds, labels)
mmae = mean_mae(preds, labels)

print(f"Mean QWK: {mqwk:.4f}")
print(f"Mean MAE: {mmae:.4f}")
print(f"Best items: {sorted(range(21), key=lambda i: piqwk[i], reverse=True)[:3]}")
print(f"Worst items: {sorted(range(21), key=lambda i: piqwk[i])[:3]}")
```