import cv2
import numpy as np
import os

def segment_fan_and_save(image_folder, output_mask_folder):
    # 创建输出文件夹
    if not os.path.exists(output_mask_folder):
        os.makedirs(output_mask_folder)

    # 获取输入文件夹中的所有图像文件
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

    for image_file in image_files:
        # 构建输入图像的完整路径
        image_path = os.path.join(image_folder, image_file)

        # 读取灰度图像
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # 使用阈值分割，将灰度图转为二值图像
        _, binary_image = cv2.threshold(image, 0.5, 255, cv2.THRESH_BINARY)

        # 寻找轮廓
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 创建一个与图像大小相同的黑色图像
        mask = np.zeros_like(image)

        # 将扇形内的轮廓绘制到掩模上
        cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

        # 构建输出 mask 图像的完整路径
        output_mask_path = os.path.join(output_mask_folder, f"{image_file}")

        # 保存 mask 图像
        cv2.imwrite(output_mask_path, mask)

if __name__ == "__main__":
    # 替换为您的输入和输出文件夹路径
    input_image_folder = "C:\Project\Pycharm\My-U-Net\TrainTarget\Training_Images"
    output_mask_folder = "C:\Project\Pycharm\My-U-Net\TrainTarget\Training_Images_mask"

    segment_fan_and_save(input_image_folder, output_mask_folder)
    print('OK')
