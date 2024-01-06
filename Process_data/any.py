import cv2
import numpy as np

def segment_fan(image_path):
    # 读取灰度图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    image[:100, :] = 0
    image[:, :200] = 0
    image[:, -200:] = 0
    image[-100:,:] = 0

    # 使用阈值分割，将灰度图转为二值图像
    _, binary_image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)

    # 寻找轮廓
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建一个与图像大小相同的黑色图像
    mask = np.zeros_like(image)

    # 将扇形内的轮廓绘制到掩模上
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

    # 将掩模应用到原始图像
    result = cv2.bitwise_and(image, mask)

    # 显示原始图像和处理后的图像
    cv2.imshow("Original Image", image)
    cv2.imshow("Result", result)
    cv2.imshow("Result", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 替换为您的图像文件路径
image_path = "/TrainTarget/Training_Images/50.jpg"
segment_fan(image_path)
