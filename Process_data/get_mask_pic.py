import json
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from PIL import Image
import numpy as np
import base64
import io

# 输入目录包含所有的JSON文件
input_directory = 'mark_label'
# 输出目录用于保存掩码图像
output_directory = 'mask_images'

# 创建输出目录，如果它不存在
os.makedirs(output_directory, exist_ok=True)

# 遍历输入目录中的所有JSON文件
for filename in os.listdir(input_directory):
    if filename.endswith('.json'):
        # 构建JSON文件的完整路径
        json_file_path = os.path.join(input_directory, filename)

        # 读取JSON文件
        with open(json_file_path, 'r') as json_file:
            mask_data = json.load(json_file)

        # 获取图像的长和宽
        image_width = mask_data['imageWidth']
        image_height = mask_data['imageHeight']

        # 3. 创建一个空白图像
        blank_image = plt.figure(figsize=(image_width / 100, image_height / 100)).add_subplot(111)
        blank_image.set_xlim(0, image_width)
        blank_image.set_ylim(image_height, 0)

        # 4. 创建一个图形集合来存储多边形掩码
        patches = []
        for shape in mask_data['shapes']:
            label = shape['label']
            points = shape['points']
            polygon = Polygon(points, closed=True, edgecolor=None, facecolor='red', label=label)
            patches.append(polygon)

        collection = PatchCollection(patches, match_original=True)

        # 5. 在图像上绘制掩码
        blank_image.add_collection(collection)
        blank_image.autoscale_view()


        # # 获取图像数据并显示图像
        # image_data = mask_data['imageData']
        # image_data = base64.b64decode(image_data)
        # image = Image.open(io.BytesIO(image_data))
        # plt.imshow(image)

        # 生成掩码图像的文件名
        mask_image_filename = os.path.splitext(filename)[0] + '_mask.png'
        mask_image_path = os.path.join(output_directory, mask_image_filename)

        # 保存掩码图像到输出目录
        plt.axis('off')
        plt.savefig(mask_image_path, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()

print("掩码图像已保存到输出目录:", output_directory)

#判断一下属于哪个标签
#读取分数放到txt？