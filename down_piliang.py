import cv2
import os

# 源文件夹和目标文件夹路径
source_folder = r"D:\Deep_learning\new\radar_ceshi\xin\monsoon-season_heavy_rainstorms\ZG101_20230615\imagecrop"
target_folder = r"D:\Deep_learning\new\radar_ceshi\xin\monsoon-season_heavy_rainstorms\ZG101_20230615\image512"
# 确保目标文件夹存在
os.makedirs(target_folder, exist_ok=True)

# 遍历源文件夹中的所有文件
for filename in os.listdir(source_folder):
    # 构建完整的文件路径
    input_image_path = os.path.join(source_folder, filename)
    output_image_path = os.path.join(target_folder, filename)

    # 检查文件是否为图像（这里简单检查扩展名，但这不是最安全的方法）
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        # 读取图像
        img = cv2.imread(input_image_path)

        # 检查图像是否读取成功
        if img is not None:
            # 调整图像尺寸为 770 x 610（注意：这可能会改变图像的宽高比）
            # 如果你希望保持宽高比，你需要先计算缩放因子，然后应用它
            # resized_img = cv2.resize(img, (770, 610))
            resized_img = cv2.resize(img, (512, 512))
            # 保存调整后的图像
            cv2.imwrite(output_image_path, resized_img)
            print(f"调整后的图像 {filename} 已保存到 {target_folder}")
        else:
            print(f"无法读取图像文件 {input_image_path}")
    else:
        # 如果文件不是图像，则跳过它（这里可以根据需要添加其他文件类型的处理）
        print(f"跳过非图像文件 {filename}")

        # 脚本执行完毕，没有需要关闭的OpenCV窗口（因为没有使用imshow）