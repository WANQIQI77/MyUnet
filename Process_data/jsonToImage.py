from PIL import Image, ImageDraw
import json
import os

# 指定JSON文件所在的文件夹路径
json_folder_path = 'path/to/your/json/folder'

# 指定输出图像的文件夹路径
output_folder_path = 'path/to/your/output/folder'

# 确保输出文件夹存在，如果不存在则创建
os.makedirs(output_folder_path, exist_ok=True)

# 遍历文件夹中的所有JSON文件
for json_filename in os.listdir(json_folder_path):
    if json_filename.endswith('.json'):
        json_filepath = os.path.join(json_folder_path, json_filename)

        # 读取JSON文件
        with open(json_filepath, 'r') as json_file:
            data = json.load(json_file)

        # 创建一张新的图像
        image = Image.new('RGB', (data['imageWidth'], data['imageHeight']), color='white')
        draw = ImageDraw.Draw(image)

        # 绘制多边形
        for shape in data['shapes']:
            label = shape['label']
            points = shape['points']
            draw.polygon(points, outline='red', fill=None)  # 设置轮廓颜色为红色，填充为无

        # 保存图像，以JSON文件名为基础，将其替换为.png扩展名
        output_filename = os.path.splitext(json_filename)[0] + '.png'
        output_filepath = os.path.join(output_folder_path, output_filename)
        image.save(output_filepath)

# 完成转换
print("转换完成")
