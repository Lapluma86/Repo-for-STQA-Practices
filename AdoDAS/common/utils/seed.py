import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 325799) -> None:
    random.seed(seed) # 固定random种子
    os.environ["PYTHONHASHSEED"] = str(seed) # 固定Python hash种子
    np.random.seed(seed) # 固定numpy种子
    torch.manual_seed(seed) # 固定PyTorch种子
    torch.cuda.manual_seed_all(seed) # 固定所有GPU的种子
    torch.backends.cudnn.deterministic = True # 确保每次运行时卷积算法的确定性
    torch.backends.cudnn.benchmark = False # 禁用cudnn的自动优化
