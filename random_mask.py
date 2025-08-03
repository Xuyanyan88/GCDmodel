import numpy as np
import random
import math
import os
import cv2  # 引入OpenCV库
from PIL import Image

# 定义图像的尺寸
height, width = 512, 512

# 确保保存路径存在
save_dir = r'D:\Deep learning\new\xi_mask'  # 请根据你的实际情况修改路径
os.makedirs(save_dir, exist_ok=True)


# 函数：生成一个随机掩码并保存
def generate_and_save_mask(index):
    # 创建一个空白的掩码数组
    mask = np.zeros((height, width), dtype=np.uint8)

    # 定义要生成的射线数量范围
    num_rays = random.randint(3, 5)

    # 定义膨胀操作使用的核大小（这将决定线条的粗细）
    kernel_size = (5, 5)  # 你可以根据需要调整这个大小

    # 生成随机射线
    for _ in range(num_rays):
        # 随机生成一个角度（0到2π之间）
        angle = random.uniform(0, 2 * math.pi)

        # 计算射线的斜率
        if angle == math.pi / 2 or angle == 3 * math.pi / 2:
            slope = float('inf')  # 垂直线
        else:
            slope = math.tan(angle)

            # 定义射线的起点为中心点
        center_x, center_y = width // 2, height // 2

        # 计算射线与图像边框的交点，并绘制在掩码上（这里简化处理，只考虑整像素点）
        x_starts, x_ends = [], []
        y_starts, y_ends = [], []

        if slope == float('inf'):  # 垂直线
            x_end = center_x
            y_starts = [0, height - 1]
            y_ends = y_starts
        elif slope == 0:  # 水平线
            y_end = center_y
            x_starts = [0, width - 1]
            x_ends = x_starts
        else:
            # 计算交点（这里为了简化，只考虑整像素交点）
            x_intercept_top = int(round((0 - center_y) * (1 / slope) + center_x))
            x_intercept_bottom = int(round((height - 1 - center_y) * (1 / slope) + center_x))
            y_intercept_left = int(round(slope * (0 - center_x) + center_y))
            y_intercept_right = int(round(slope * (width - 1 - center_x) + center_y))

            # 根据斜率决定遍历方向并绘制射线（整像素点）
            if abs(slope) < 1:  # 平缓的斜线
                x_starts = list(range(width))
                for x in x_starts:
                    y = int(round(slope * (x - center_x) + center_y))
                    if 0 <= y < height:
                        mask[y, x] = 1
            else:  # 陡峭的斜线
                y_starts = list(range(height))
                for y in y_starts:
                    x = int(round((y - center_y) / slope + center_x))
                    if 0 <= x < width:
                        mask[y, x] = 1

                        # 由于我们之前计算的是整像素交点，并且可能漏掉了一些点，
            # 所以这里我们不再使用x_ends和y_ends，而是直接使用mask数组。
            # 但为了保持函数结构的一致性，我们仍然保留这些变量（为空列表）。

        # 注意：上面的绘制方法可能不是最优的，因为它可能会在某些情况下漏掉一些像素点。
        # 一个更准确的方法是使用Bresenham算法或类似的直线绘制算法。
        # 但为了简化，我们在这里使用这种方法，并通过膨胀来使线条变粗。

    # 使用OpenCV的膨胀操作使线条变粗
    dilated_mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size), iterations=1)

    # 将OpenCV图像转换为PIL图像并保存
    pil_image = Image.fromarray(dilated_mask * 255).convert('L')  # 转换为灰度图像
    file_name = f"bin_xi{index:04d}.png"
    save_path = os.path.join(save_dir, file_name)
    pil_image.save(save_path)


# 生成1000张随机掩码
for i in range(1, 501):
    generate_and_save_mask(i)

print("1000张随机粗线条掩码已生成并保存。")