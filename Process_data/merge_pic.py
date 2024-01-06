'''
将图片合并为7*H*W的numpy数组
'''
from PIL import Image
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json

# 输入目录包含所有的JSON文件
input_directory = 'pre_merge_pic'
# 输出目录用于保存掩码图像
output_directory = 'merge_images'

# 创建输出目录，如果它不存在
os.makedirs(output_directory, exist_ok=True)
pic_num = 40

all_img_data = [[[] for _ in range(7)] for _ in range(pic_num)]
'''
#文件夹XXX_1.png--->1.png
class_file_dict = {}

# 遍历每个文件夹
for folder_num in range(1, 8):
    folder_name = f'{input_directory}/{folder_num}'
    for filename in os.listdir(folder_name):
        # 获取图片的数字部分
        image_number = int(filename.split("_")[1].split(".")[0])
        class_name = filename.split('_')[0]
        image_path = os.path.join(folder_name, filename)
        os.rename(image_path, f'{folder_name}/{image_number}.png')
        class_file_dict[folder_num] = class_name

out_file = open("class_file.json", "w")
json.dump(class_file_dict, out_file)
out_file.close()
'''

for folder_num in range(1, 8):
    folder_name = f'{input_directory}/{folder_num}'
    filename = os.listdir(folder_name)
    for pic_no in range(pic_num):
        if f'{pic_no}.png' in filename:
            image_path = os.path.join(folder_name, f'{pic_no}.png')
            # 打开图像并添加到相应数字的列表中
            img = Image.open(image_path).convert('L')
            img_data = np.array(img)
            img_data = cv2.resize(img_data, (512, 512))
            all_img_data[pic_no - 1][folder_num - 1].append(img_data)
        else:
            img_data = np.ones((512, 512))
            all_img_data[pic_no - 1][folder_num - 1].append(img_data)

all_img_data = np.array(all_img_data).squeeze()
# 存储numpy文件
for i in range(1, pic_num + 1):
    np.save(f'TrainTarget/training_numpy/{i}.npy', all_img_data[i-1])

# 显示一个试试看
plt.imshow(all_img_data[0][0])
plt.show()

# 关闭所有图像文件
for folder_num in range(1, 8):
    folder_name = f'{input_directory}/{folder_num}'
    for filename in os.listdir(folder_name):
        if filename.endswith(".png"):
            img = Image.open(os.path.join(folder_name, filename))
            img.close()

print('图像合并完成')
