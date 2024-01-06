# -*- coding: utf-8 -*-

import os
from tqdm import tqdm
from utils.utils_Metrics import Compute_mIoU, show_results
import glob
import numpy as np
import torch
import os
import cv2
from model.unet_model import UNet
from utils.Basic_Function import Get_Average_From_Matrix2
from PIL import Image

'''
 # 类别颜色映射
    class_colors = {
        0: (255, 255, 255),  # 白色
        1: (255, 165, 0),  # 橙色 胎盘位置
        2: (128, 128, 128),  # 灰色  胎盘厚度
        3: (0, 128, 0),  # 绿色 回声带
        4: (0, 0, 255),  # 蓝色 膀胱线
        5: (255, 255, 0),  # 黄色 胎盘陷窝
        6: (255, 0, 0),  # 红色 血瘤
        7: (128, 0, 128),  # 紫色 宫颈形态
        8: (0, 0, 0)  # 黑色
    }
'''

def Calculate_mIOU(epoches, test_dir="TrainTarget/Test_Images/",
                   pred_dir="TrainTarget/results/",
                   gt_dir="TrainTarget/Test_numpy/",
                   mask_dir="TrainTarget/Test_Images_mask"):

    miou_mode = 2
    num_classes = 8
    name_classes = ["_background_", "weizhi","houdu","huishengdai","pangguangxian","xianwo","xueliu","gongjing"]

    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)

        print("Load model.")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net = UNet(n_channels=3, n_classes=7)
        net.to(device=device)
        net.load_state_dict(torch.load('epoch={}_model_Unet_8_1e-6.pth'.format(str(epoches)), map_location=device))  # todo

        # 测试模式
        net.eval()
        print("Load model done.")

        img_names = os.listdir(test_dir)
        image_ids = [image_name.split(".")[0] for image_name in img_names]

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path = os.path.join(test_dir, image_id + ".jpg")
            img = cv2.imread(image_path)
            print(image_path)
            origin_shape = img.shape

            img = cv2.resize(img, (512, 512))
            # 转为batch为1，通道为3，大小为512*512的数组
            img = img.reshape(1, 3, img.shape[0], img.shape[1])
            img_tensor = torch.from_numpy(img)
            img_tensor = img_tensor.to(device=device, dtype=torch.float32)
            with torch.no_grad():
                output = net(img_tensor)  # 添加 batch 维度
                # 获取每个像素的类别概率
            class_probabilities = torch.nn.functional.softmax(output, dim=1)
            # 转换为numpy数组，每个通道对应一个类别的概率
            pred = class_probabilities.squeeze().cpu().numpy()
            a = Get_Average_From_Matrix2(pred[0])
            pred = (pred > 0.95).astype(np.uint8)
            np.save(os.path.join('TrainTarget/results_numpy', image_id + ".npy"), pred)

            threshold = 0.5
            result_image = np.ones((512, 512, 3), dtype=np.uint8)*255
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
            # 获取 mask 图片中颜色背景的位置（黑色）
            white_pixels = np.column_stack(np.where(np.all(mask == [0, 0, 0], axis=-1)))
            # 在分割图片对应位置保留颜色
            result_image[white_pixels[:, 0], white_pixels[:, 1]] = (0, 0, 0)
            result_image = cv2.resize(result_image, (origin_shape[1], origin_shape[0]), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(pred_dir, image_id + ".png"), result_image)
        print("Get predict result done.")

    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        print(gt_dir)
        print(pred_dir)
        print(num_classes)
        print(name_classes)
        hist, IoUs, PA_Recall, Precision = Compute_mIoU(gt_dir, pred_dir, image_ids, num_classes, name_classes)
        print("Get miou done.")
        miou_out_path = "TrainTarget/results/"
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)


if __name__ == '__main__':
    Calculate_mIOU(100)
