# -*- coding: utf-8 -*-
"""
Eyecandies 数据集预处理脚本（一次性运行）。

Eyecandies 原始目录结构较复杂：每个物体目录下的 train/test/val 里，每个样本
都有 17 张辅助图（多视角渲染 + depth + normals + mask 等），文件名形如
``000_image_4.png``、``000_depth.png``、``000_normals.png``、``000_mask.png``。

本脚本的作用：
    - 将 Eyecandies 原始目录整理成与 MVTec 3D-AD 兼容的 "good / bad" 目录结构，
      方便 ``dataset/eyecandies.py`` 直接按 rgb/depth/normals/gt 四个子目录读取。
    - train/val 只有正常样本 -> 全部放入 ``good``；
    - test 根据 mask 是否全零拆分到 ``good`` / ``bad``，bad 的保存 mask 作为 gt。

使用方法：
    1. 修改 ``dataset_path`` 为原始 Eyecandies 解压目录；
    2. 修改 ``target_dir`` 为目标输出目录（需要事先删除已存在同名目录）；
    3. 直接 ``python preprocess_eyecandies.py`` 运行。

注意：``os.mkdir`` 不会自动创建多级目录，若目标目录已存在会报错，
属于"只需一次"的工具脚本。
"""

import os
from shutil import copyfile
import cv2
import numpy as np


if __name__ == '__main__':

    # --------- 1. 路径配置：根据实际环境修改 ---------
    dataset_path = 'F:\TRD\data\Eyecandies'                     # Eyecandies 原始目录
    target_dir = 'F:\TRD\data\Eyecandies_preprocessed'          # 预处理后输出目录
    os.mkdir(target_dir)

    # --------- 2. 遍历每个类别 ---------
    categories_list = os.listdir(dataset_path)
    print(categories_list)

    for category_dir in categories_list:
        category_root_path = os.path.join(dataset_path, category_dir)

        print(type(category_root_path))

        # Eyecandies 的 train/test/val 样本都统一放在各自的 ``data`` 子目录下
        category_train_path = os.path.join(category_root_path, 'train/data')
        category_test_path = os.path.join(category_root_path, 'test_public/data')
        print(category_train_path)
        print(category_test_path)

        category_val_path = os.path.join(category_root_path, 'val/data')

        # --------- 3. 在目标目录下准备好与 MVTec 3D 对齐的文件夹结构 ---------
        category_target_path = os.path.join(target_dir, category_dir)
        os.mkdir(category_target_path)

        # train/good/{rgb, normals, depth}
        os.mkdir(os.path.join(category_target_path, 'train'))
        category_target_train_good_path = os.path.join(category_target_path, 'train/good')
        category_target_train_good_rgb_path = os.path.join(category_target_train_good_path, 'rgb')
        category_target_train_good_normal_path = os.path.join(category_target_train_good_path, 'normals')
        category_target_train_good_depth_path = os.path.join(category_target_train_good_path, 'depth')
        os.mkdir(category_target_train_good_path)
        os.mkdir(category_target_train_good_rgb_path)
        os.mkdir(category_target_train_good_normal_path)
        os.mkdir(category_target_train_good_depth_path)

        # test/good 与 test/bad，两者都有 rgb/normals/depth/gt 四个子目录
        os.mkdir(os.path.join(category_target_path, 'test'))
        category_target_test_good_path = os.path.join(category_target_path, 'test/good')
        category_target_test_good_rgb_path = os.path.join(category_target_test_good_path, 'rgb')
        category_target_test_good_normal_path = os.path.join(category_target_test_good_path, 'normals')
        category_target_test_good_depth_path = os.path.join(category_target_test_good_path, 'depth')
        category_target_test_good_gt_path = os.path.join(category_target_test_good_path, 'gt')
        os.mkdir(category_target_test_good_path)
        os.mkdir(category_target_test_good_rgb_path)
        os.mkdir(category_target_test_good_normal_path)
        os.mkdir(category_target_test_good_depth_path)
        os.mkdir(category_target_test_good_gt_path)
        category_target_test_bad_path = os.path.join(category_target_path, 'test/bad')
        category_target_test_bad_rgb_path = os.path.join(category_target_test_bad_path, 'rgb')
        category_target_test_bad_normal_path = os.path.join(category_target_test_bad_path, 'normals')
        category_target_test_bad_depth_path = os.path.join(category_target_test_bad_path, 'depth')
        category_target_test_bad_gt_path = os.path.join(category_target_test_bad_path, 'gt')
        os.mkdir(category_target_test_bad_path)
        os.mkdir(category_target_test_bad_rgb_path)
        os.mkdir(category_target_test_bad_normal_path)
        os.mkdir(category_target_test_bad_depth_path)
        os.mkdir(category_target_test_bad_gt_path)

        # --------- 4. 训练集：每个样本有 17 个文件，只取其中 3 张（rgb 视角 4 / depth / normals）---------
        category_train_files = os.listdir(category_train_path)
        num_train_files = len(category_train_files) // 17   # 真实样本数
        for i in range(0, num_train_files):
            # image_4 是一个选定的光照视角（Eyecandies 每个样本有多个渲染视角）
            copyfile(os.path.join(category_train_path, str(i).zfill(3)+'_image_4.png'),
                     os.path.join(category_target_train_good_rgb_path, str(i).zfill(3)+'.png'))
            copyfile(os.path.join(category_train_path, str(i).zfill(3) + '_depth.png'),
                     os.path.join(category_target_train_good_depth_path, str(i).zfill(3) + '.png'))
            copyfile(os.path.join(category_train_path, str(i).zfill(3) + '_normals.png'),
                     os.path.join(category_target_train_good_normal_path, str(i).zfill(3) + '.png'))

        # --------- 5. 测试集：根据 mask 判断 good/bad 并相应归档 ---------
        category_test_files = os.listdir(category_test_path)
        num_test_files = len(category_test_files) // 17
        for i in range(0, num_test_files):
            # 注意：测试集文件名是 2 位数字前缀（而不是 3 位），与训练集命名略有不同
            mask = cv2.imread(os.path.join(category_test_path, str(i).zfill(2)+'_mask.png'))
            if np.any(mask):
                # mask 非全零 -> 异常样本，放入 test/bad（此处未保存 gt，可按需补充）
                copyfile(os.path.join(category_test_path, str(i).zfill(2)+'_image_4.png'),
                         os.path.join(category_target_test_bad_rgb_path, str(i).zfill(3)+'.png'))
                copyfile(os.path.join(category_test_path, str(i).zfill(2) + '_depth.png'),
                         os.path.join(category_target_test_bad_depth_path, str(i).zfill(3) + '.png'))
                copyfile(os.path.join(category_test_path, str(i).zfill(2) + '_normals.png'),
                         os.path.join(category_target_test_bad_normal_path, str(i).zfill(3) + '.png'))
            else:
                # mask 全零 -> 正常样本，放入 test/good。gt 保存为空 mask 便于评估统一处理
                cv2.imwrite(os.path.join(category_target_test_good_gt_path, str(i).zfill(3)+'.png'), mask)
                copyfile(os.path.join(category_test_path, str(i).zfill(2)+'_image_4.png'),
                         os.path.join(category_target_test_good_rgb_path, str(i).zfill(3)+'.png'))
                copyfile(os.path.join(category_test_path, str(i).zfill(2) + '_normals.png'),
                         os.path.join(category_target_test_good_normal_path, str(i).zfill(3) + '.png'))
                copyfile(os.path.join(category_test_path, str(i).zfill(2) + '_depth.png'),
                         os.path.join(category_target_test_good_depth_path, str(i).zfill(3) + '.png'))

        # --------- 6. 验证集：全部视为 good ---------
        os.mkdir(os.path.join(category_target_path, 'validation'))
        category_target_val_good_path = os.path.join(category_target_path, 'validation/good')
        category_target_val_good_rgb_path = os.path.join(category_target_val_good_path, 'rgb')
        category_target_val_good_normal_path = os.path.join(category_target_val_good_path, 'normals')
        category_target_val_good_depth_path = os.path.join(category_target_val_good_path, 'depth')
        os.mkdir(category_target_val_good_path)
        os.mkdir(category_target_val_good_rgb_path)
        os.mkdir(category_target_val_good_normal_path)
        os.mkdir(category_target_val_good_depth_path)

        category_val_files = os.listdir(category_val_path)
        num_val_files = len(category_val_files) // 17
        for i in range(0, num_val_files):
            copyfile(os.path.join(category_val_path, str(i).zfill(2) + '_image_4.png'),
                     os.path.join(category_target_val_good_rgb_path, str(i).zfill(3) + '.png'))
            copyfile(os.path.join(category_val_path, str(i).zfill(2) + '_depth.png'),
                     os.path.join(category_target_val_good_depth_path, str(i).zfill(3) + '.png'))
            copyfile(os.path.join(category_val_path, str(i).zfill(2) + '_normals.png'),
                     os.path.join(category_target_val_good_normal_path, str(i).zfill(3) + '.png'))
