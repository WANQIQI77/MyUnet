import numpy as np
import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import random


class ISBI_Loader(Dataset):
    def __init__(self, data_path):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        # self.files = [f'training_numpy/{i}.npy' for i in range(1, 41)]# npy文件路径
        # self.files = [f'{i}.npy' for i in range(25, 1600)]
        self.imgs_path = glob.glob(os.path.join(data_path, 'Training_Images/*.jpg'))
        self.numpy_path = glob.glob(os.path.join(data_path, 'Training_numpy/*.npy'))
        ''''''
        # self.files = [f'training_numpy/{i}.npy' for i in range(1, 41)]# npy文件路径

    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip

    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.imgs_path[index]
        label_path = self.numpy_path[index]

        # 根据image_path生成label_path
        ''''''
        # 删除
        # label_path = image_path.replace('Training_Images', 'Training_Labels')
        # label_path = label_path.replace('.jpg', '.png')  # todo 更新标签文件的逻辑

        # 读取训练图片和标签图片
        image = cv2.imread(image_path)
        label = np.load(label_path)

        # label = cv2.imread(label_path)
        image = cv2.resize(image, (512, 512))
        # label = cv2.resize(label, (512, 512), interpolation=cv2.INTER_NEAREST)
        # 将数据转为单通道的图片
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        '''
        # 图片加mask
        mask = cv2.imread('C:\Project\Pycharm\My-U-Net\mask_images\\1_mask.png')  # 替换为您的图片路径
        mask = cv2.resize(mask, (512, 512))
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        # 获取mask图片中颜色白色的位置
        white_pixels = np.where(mask_gray == 255)
        image[white_pixels] = 0
        '''
        # # 处理标签，将像素值为255的改为1
        # if label.max() > 1:
        #     label = label / 255
        # 随机进行数据增强，为2时不做处理
        flipCode = random.choice([-1, 0, 1, 2])
        if flipCode != 2:
            image = self.augment(image, flipCode)
            label = self.augment(label, flipCode)
        image = image.reshape(3, image.shape[0], image.shape[1])
        label = label.reshape(7, label.shape[1], label.shape[2])

        ''''''
        # 转成tensor
        image = torch.from_numpy(image)
        label = torch.from_numpy(label)

        return image, label

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)


if __name__ == "__main__":
    isbi_dataset = ISBI_Loader("C:/Project/Pycharm/My-U-Net/TrainTarget/")
    print("数据个数：", len(isbi_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=32,
                                               shuffle=True)
    for image, label in train_loader:
        print(image.shape)
