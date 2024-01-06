# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Project Name: unet
File Name: Gen_Split.py
Author: chenming
Create Date: 2022/2/6
Description：
-------------------------------------------------
"""
# 去除seg
import os
import os.path as osp
import shutil


def Remove_Seg(folder_path):
    img_files = os.listdir(folder_path)
    for img_file in img_files:
        img_path = osp.join(folder_path, img_file)
        traget_path = img_path.replace('_Segmentation.png', '.png')
        shutil.copy(img_path, traget_path)


if __name__ == '__main__':
    Remove_Seg("F:/xxxxxxxxxx/xianyu/skin/Test_GroundTruth")
    Remove_Seg("F:/xxxxxxxxxx/xianyu/skin/Training_GroundTruth")