# -*- coding: utf-8 -*-
"""
Teacher-Reverse-Distillation（TRD）框架使用的各种损失函数。

一个典型的 RD 类异常检测训练目标：让 **学生（解码器）** 对齐 **教师（编码器）** 的多尺度特征。
本文件提供三组损失：
    1. 逐元素距离：
        - ``loss_l2``        : MSE，对特征图做逐元素平方差。
        - ``loss_distil``    : 先把每个尺度 flatten 成向量再做 1 - 余弦相似度。
        - ``loss_distil_p``  : 保持 H×W 空间维度的余弦距离，对每个像素求平均。
    2. 像素级相似度对齐（更细粒度）：
        - ``loss_distil_pixel`` : 在每个 batch 内构造 NxN 像素相似度矩阵，
          要求学生与教师的"像素-像素"关系保持一致。
    3. 辅助函数：
        - ``calculate_pixel_similarity`` / ``_f`` / ``_f_2`` : 计算 NxN 余弦相似度矩阵。

文件末尾保留了一大段 ``loss_pixel_attention`` 的注释代码，为历史实验遗留，
当前训练脚本未启用。
"""

import torch
import torch.nn.functional as F


def loss_l2(feature_s, feature_t):
    """逐元素 MSE 蒸馏损失。

    feature_s: 学生多尺度特征列表（已与教师对齐尺寸）
    feature_t: 教师多尺度特征列表
    """
    loss_type = torch.nn.MSELoss()
    loss = 0.0
    for i in range(len(feature_s)):
        loss_i = loss_type(feature_s[i], feature_t[i])
        loss += loss_i

    return loss


def loss_distil(feature_s, feature_t):
    """"向量级"余弦蒸馏：把每个尺度 flatten 为 (B, C*H*W)，再做余弦相似度。

    优点：尺度不变、对整体结构一致性敏感；
    缺点：丢失空间信息，对局部异常不够敏感。
    """
    loss_type = torch.nn.CosineSimilarity()
    loss = 0.0
    for i in range(len(feature_s)):
        # flatten 每个样本到 1D 向量，CosineSimilarity 默认沿 dim=1 计算
        loss_i = torch.mean(1 - loss_type(
            feature_s[i].reshape(feature_s[i].shape[0], -1),
            feature_t[i].reshape(feature_t[i].shape[0], -1)))
        loss += loss_i

    return loss


def loss_distil_p(feature_s, feature_t):
    """逐像素余弦蒸馏（保留空间信息）。

    CosineSimilarity 的默认 dim=1 对应通道维，所以相当于在每个像素位置上
    把学生/教师的通道向量做余弦相似度，再对所有像素求均值。
    """
    loss_type = torch.nn.CosineSimilarity()
    loss = 0.0
    for i in range(len(feature_s)):
        cos = 1 - loss_type(feature_s[i], feature_t[i])  # (B, H, W)
        loss_i = torch.mean(cos)
        loss += loss_i

    return loss


def loss_distil_pixel(feature_s, feature_t):
    """像素-像素关系对齐损失（更细粒度的 SP/Relational Distillation 思路）。

    思路：在每个特征图上构造像素间的 NxN 余弦相似度矩阵（相当于自注意力 attention map），
    要求学生和教师的 attention map 尽量一致（MSE）。
    为了控制显存：前两级特征先做 AvgPool 下采样（分别 4×、2×），只有最深的第三级
    在原始分辨率下计算。
    """
    loss = 0.0
    loss_type = torch.nn.MSELoss()
    # 对浅层特征做不同倍率的池化：第 0 尺度池化 4 倍，第 1 尺度池化 2 倍，
    # 第 2 尺度不池化（因为它已经是最低分辨率）
    pool_s = torch.nn.ModuleList([
        torch.nn.AvgPool2d(4, stride=4),
        torch.nn.AvgPool2d(2, stride=2)
    ])
    pool_t = torch.nn.ModuleList([
        torch.nn.AvgPool2d(4, stride=4),
        torch.nn.AvgPool2d(2, stride=2)
    ])

    for i in range(len(feature_s)):

        f_s = feature_s[i]
        f_t = feature_t[i]

        # 历史实验：对第 0 尺度按 32x32 patch 分块计算相似度矩阵（为降低内存）
        # 当前默认改为统一 AvgPool 下采样，代码更简洁
        if i == 0 or i == 1:
            f_s = pool_s[i](f_s)
            f_t = pool_t[i](f_t)

        # 计算学生/教师各自的 NxN 余弦相似度矩阵
        cos_sim_s, cos_sim_t = calculate_pixel_similarity(f_s, f_t)

        # 要求两个相似度矩阵尽量一致
        loss += loss_type(cos_sim_s, cos_sim_t)

    return loss


def calculate_pixel_similarity(f_s, f_t):
    """分别计算学生 f_s 与教师 f_t 的像素间 NxN 余弦相似度矩阵。

    返回：
        cos_sim_s : (B, N, N) 学生的自相似度矩阵
        cos_sim_t : (B, N, N) 教师的自相似度矩阵
    这里 N = H*W。注意：显存消耗为 O(B*N^2)，对深层特征需先下采样。
    """
    # ---- 学生 ----
    B, C, H, W = f_s.shape
    f_s = f_s.contiguous().view(B, C, -1)  # (B, C, N)
    f_s = f_s.transpose(1, 2)              # (B, N, C)
    cos_sim_s = torch.empty(f_s.shape[0], f_s.shape[1], f_s.shape[1]).to(f_s.device)
    # 按 batch 循环计算，避免一次性构造 (B, N, N, C) 的巨大中间张量
    for batch in range(f_s.shape[0]):
        cos_sim_s[batch] = F.cosine_similarity(
            f_s.unsqueeze(2)[batch:batch + 1, :, :, :],  # (1, N, 1, C)
            f_s.unsqueeze(1)[batch:batch + 1, :, :, :],  # (1, 1, N, C)
            dim=-1,
        )

    # ---- 教师（同样操作）----
    B, C, H, W = f_t.shape
    f_t = f_t.contiguous().view(B, C, -1)
    f_t = f_t.transpose(1, 2)
    cos_sim_t = torch.empty(f_t.shape[0], f_t.shape[1], f_t.shape[1]).to(f_s.device)
    for batch in range(f_t.shape[0]):
        cos_sim_t[batch] = F.cosine_similarity(
            f_t.unsqueeze(2)[batch:batch + 1, :, :, :],
            f_t.unsqueeze(1)[batch:batch + 1, :, :, :],
            dim=-1,
        )

    return cos_sim_s, cos_sim_t


def calculate_pixel_similarity_f(f):
    """单张特征图的像素自相似度矩阵（便于在注释掉的实验代码中使用）。"""
    B, C, H, W = f.shape
    f = f.contiguous().view(B, C, -1)
    f = f.transpose(1, 2)
    cos_sim = torch.empty(f.shape[0], f.shape[1], f.shape[1]).to(f.device)
    for batch in range(f.shape[0]):
        cos_sim[batch] = F.cosine_similarity(f.unsqueeze(2)[batch:batch + 1, :, :, :],
                                               f.unsqueeze(1)[batch:batch + 1, :, :, :], dim=-1)

    return cos_sim


def calculate_pixel_similarity_f_2(f_s, f_t):
    """学生像素 vs 教师像素的 NxN 交叉相似度矩阵（非对称）。"""
    B, C, H, W = f_s.shape
    f_s = f_s.contiguous().view(B, C, -1)
    f_s = f_s.transpose(1, 2)
    f_t = f_t.contiguous().view(B, C, -1)
    f_t = f_t.transpose(1, 2)
    cos_sim = torch.empty(f_s.shape[0], f_s.shape[1], f_s.shape[1]).to(f_s.device)
    for batch in range(f_s.shape[0]):
        cos_sim[batch] = F.cosine_similarity(f_s.unsqueeze(2)[batch:batch + 1, :, :, :],
                                               f_t.unsqueeze(1)[batch:batch + 1, :, :, :], dim=-1)

    return cos_sim


# ==================== 历史实验代码（保留注释）====================
# ``loss_pixel_attention`` 是更复杂的"按 16x16 patch 分块 + 可学习权重"的
# 像素注意力蒸馏方案，当前训练管线未启用，保留以便日后做消融对比。
# def loss_pixel_attention(feature_s, feature_t):
#     ...（原实现省略，见 git 历史）
#     loss = 0.0
#     loss_type = torch.nn.L1Loss()
#
#     for i in range(len(feature_s)):
#
#         f_s = feature_s[i]
#         f_t = feature_t[i]
#
#         if i == 0 or i == 1 or i==2:
#             # continue
#             loss_tmp = 0.0
#
#             # print(f_s.shape)
#
#             # f_s = self.pool_s[i](feature_s[i])
#             # f_t = self.pool_t[i](feature_t[i])
#
#             B, C, H, W = f_s.shape
#             # print(f_s.shape)
#             for m in range(H // 16):
#                 for n in range(W // 16):
#                     f_s_patch = f_s[:, :, m * 16:(m + 1) * 16, n * 16:(n + 1) * 16]
#                     # print(f_s_patch.shape)
#                     f_t_patch = f_t[:, :, m * 16:(m + 1) * 16, n * 16:(n + 1) * 16]
#
#                     # cos_sim_st = self.calculate_pixel_similarity_f_2(f_s_patch, f_t_patch)
#                     # cos_sim_t = self.calculate_pixel_similarity_f(f_t_patch)
#                     #
#                     # loss_tmp += torch.mean((1 - cos_sim_st) * cos_sim_t)
#
#                     cos_sim_st = calculate_pixel_similarity_f_2(f_s_patch, f_t_patch)
#                     cos_sim_t = calculate_pixel_similarity_f(f_t_patch)
#                     cos_sim_s = calculate_pixel_similarity_f(f_s_patch)
#
#                     loss_tmp += loss_type(cos_sim_s, cos_sim_t)
#                     with torch.no_grad():
#                         loss_w = 1 - torch.abs(cos_sim_t - cos_sim_s)
#                         # print(loss_w)
#                     # loss_tmp += torch.mean((1 - cos_sim_st) * cos_sim_t * loss_w)
#                     loss_tmp += torch.mean(torch.abs(cos_sim_st - cos_sim_t) * loss_w)
#
#             # print(H // 32)
#             loss_tmp = loss_tmp / ((H // 16) * (W // 16))
#             # print(loss_tmp)
#
#             loss += F.relu(self.weight[i]) * loss_tmp
#             # loss += loss_tmp
#
#             continue
#
#         # cos_sim_st = self.calculate_pixel_similarity_f_2(f_s, f_t)
#         # cos_sim_t = self.calculate_pixel_similarity_f(f_t)
#         #
#         # loss += torch.mean((1-cos_sim_st) * cos_sim_t)
#
#         cos_sim_st = calculate_pixel_similarity_f_2(f_s, f_t)
#         cos_sim_t = calculate_pixel_similarity_f(f_t)
#         cos_sim_s = calculate_pixel_similarity_f(f_s)
#
#         loss_tmp = loss_type(cos_sim_s, cos_sim_t)
#         with torch.no_grad():
#             loss_w = 1 - torch.abs(cos_sim_t - cos_sim_s)
#             # print(loss_w)
#         loss_tmp += torch.mean(torch.abs(cos_sim_st - cos_sim_t) * loss_w)
#         # loss += loss_tmp
#         loss += F.relu(weight[i]) * loss_tmp
#
#     return loss