# -*- coding: utf-8 -*-
from PIL import Image
from matplotlib.colors import ListedColormap
from tqdm import tqdm
from utils.utils_Metrics import Compute_mIoU, show_results
import glob
import numpy as np
import torch
import os
import cv2
from model.unet_model import UNet
from utils.Basic_Function import Get_Average_From_Matrix2
import matplotlib.pyplot as plt


def Calculate_mIOU(epoches, test_dir="TrainTarget/Test_Images/",
                   pred_dir="TrainTarget/Test/",
                   mask_dir="TrainTarget/Test_Images_mask/"):
        # 加载模型
        print("Load model.")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net = UNet(n_channels=3, n_classes=7)
        net.to(device=device)
        net.load_state_dict(
            torch.load('epoch={}_model_UNet_8_1e-6.pth'.format(str(epoches)), map_location=device))  # todo
        # 测试模式
        net.eval()  # 评估模式
        print("Load model done.")

        img_names = os.listdir(test_dir)
        image_ids = [image_name.split(".")[0] for image_name in img_names]

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path = os.path.join(test_dir, image_id + ".jpg")
            img = cv2.imread(image_path)
            origin_shape = img.shape
            pred = np.load(os.path.join('TrainTarget/Test_numpy', image_id + ".npy"))
            threshold = 0.5
            result_image = np.ones((512, 512, 3), dtype=np.uint8) * 255
            for i in range(7):
                channel = pred[i]
                color = (255, 255, 255)  # 默认为白色
                if np.max(channel) > threshold:
                    if i == 0:
                        color = (0, 165, 255)  # 橙色
                    elif i == 1:
                        color = (128, 128, 128)  # 灰色
                    elif i == 2:
                        color = (0, 255, 0)  # 绿色
                    elif i == 3:
                        color = (255, 0, 0)  # 蓝色
                    elif i == 4:
                        color = (0, 255, 255)  # 黄色
                    elif i == 5:
                        color = (0, 0, 255)  # 红色
                    elif i == 6:
                        color = (128, 0, 128)  # 紫色
                # 根据通道值设置图像对应位置的颜色
                result_image[np.where(channel > threshold)] = color
            # 叠加背景
            # 读取要叠加的图片
            mask_path = os.path.join(mask_dir, image_id + ".jpg")
            mask = cv2.imread(mask_path)  # 替换为您的图片路径
            mask = cv2.resize(mask, (512, 512))
            cv2.imwrite(os.path.join(pred_dir, image_id + ".png"), result_image)
            # 获取 mask 图片中颜色背景的位置（黑色）
            white_pixels = np.column_stack(np.where(np.all(mask == [0, 0, 0], axis=-1)))
            # 在分割图片对应位置保留颜色
            result_image[white_pixels[:, 0], white_pixels[:, 1]] = (0, 0, 0)
            result_image = cv2.resize(result_image, (origin_shape[1], origin_shape[0]), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(pred_dir, image_id + ".png"), result_image)


if __name__ == '__main__':
    Calculate_mIOU(100)
