import os
import cv2
import numpy as np

def process_images(input_folder, output_folder):
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取输入文件夹中的所有图像文件
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

    for image_file in image_files:
        # 构建输入图像的完整路径
        input_path = os.path.join(input_folder, image_file)

        # 读取彩色图像
        image = cv2.imread(input_path)

        # 将前75行和左右200列的像素设置为黑色
        image[:100, :] = 0
        image[:, :200] = 0
        image[:, -200:] = 0
        image[-100:, :] = 0

        # 构建输出图像的完整路径
        output_path = os.path.join(output_folder, image_file)

        # 保存处理后的图像
        cv2.imwrite(output_path, image)

if __name__ == "__main__":
    # 替换为您的输入和输出文件夹路径
    input_folder = "C:\Project\Pycharm\My-U-Net\TrainTarget\Training_Images_original"
    output_folder = "C:\Project\Pycharm\My-U-Net\TrainTarget\Training_Images"

    process_images(input_folder, output_folder)
    print('OK')
