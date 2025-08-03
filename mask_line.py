import cv2
import numpy as np
import os

# 读取图像
input_image_path = r"D:\Deep_learning\radar\ctsdg_radar_2\test\irr_case64\image3\ZG064_202303270115.png"
name = input_image_path.split('\\')[-1]
mask_dir = r"D:\Deep_learning\radar\ctsdg_radar_2\test\irr_case64\mask3"
output_dir = r"D:\Deep_learning\radar\ctsdg_radar_2\test\irr_case64\posun3"

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)
os.makedirs(mask_dir, exist_ok=True)

output_image_path = os.path.join(output_dir, name)
mask_output_path = os.path.join(mask_dir, name)
print("输出图像路径:", output_image_path)
print("掩码图像路径:", mask_output_path)

img = cv2.imread(input_image_path)

# 检查图像是否读取成功
if img is None:
    print(f"无法读取图像文件 {input_image_path}")
    exit()

# 创建一个副本用于涂改和掩码图像
img_copy = img.copy()
mask = np.zeros_like(img)  # 创建一个全黑的掩码图像
drawing = False  # 是否正在绘制
start_point = None  # 初始化起点坐标
brush_size = 8  # 初始画笔大小
history_img = []  # 图像修改历史
history_mask = []  # 掩码修改历史

# 保存初始状态
history_img.append(img_copy.copy())
history_mask.append(mask.copy())


# 定义鼠标回调函数
def draw_line(event, x, y, flags, param):
    global start_point, drawing, img_copy, mask, brush_size

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing and start_point:
            cv2.line(img_copy, start_point, (x, y), (0, 0, 0), brush_size)
            cv2.line(mask, start_point, (x, y), (255, 255, 255), brush_size)
            start_point = (x, y)  # 更新起点为当前点

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        # 保存当前状态到历史
        history_img.append(img_copy.copy())
        history_mask.append(mask.copy())
        # 限制历史记录数量，防止占用过多内存
        if len(history_img) > 50:
            history_img.pop(0)
            history_mask.pop(0)


# 创建一个窗口并将鼠标回调函数绑定到窗口
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_line)

print("\n操作说明:")
print("  q: 保存并退出")
print("  ESC: 不保存退出")
print("  -: 缩小画笔大小")
print("  =: 放大画笔大小")
print("  c: 撤销上一步操作")
print("  r: 清除当前掩码")

while True:
    # 显示当前画笔大小
    display_img = img_copy.copy()
    cv2.putText(display_img, f"画笔大小: {brush_size}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display_img, "q:保存并退出 | ESC:不保存退出 | -:缩小画笔 | =:放大画笔 | c:撤销 | r:清除",
                (10, display_img.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow('image', display_img)
    k = cv2.waitKey(1) & 0xFF

    if k == ord('q'):  # 按 q 键保存并退出
        cv2.imwrite(output_image_path, img_copy)
        cv2.imwrite(mask_output_path, mask)
        print(f"图像已保存到 {output_image_path}")
        print(f"掩码图像已保存到 {mask_output_path}")
        break

    elif k == 27:  # ESC 键不保存退出
        print("未保存，程序已退出")
        break

    elif k == ord('-'):  # 缩小画笔大小
        brush_size = max(1, brush_size - 1)
        print(f"画笔大小已调整为: {brush_size}")

    elif k == ord('='):  # 放大画笔大小
        brush_size = min(50, brush_size + 1)
        print(f"画笔大小已调整为: {brush_size}")

    elif k == ord('c'):  # 撤销上一步操作
        if len(history_img) > 1:
            history_img.pop()
            history_mask.pop()
            img_copy = history_img[-1].copy()
            mask = history_mask[-1].copy()
            print("已撤销上一步操作")
        else:
            print("没有可撤销的操作")

    elif k == ord('r'):  # 清除当前掩码
        img_copy = img.copy()
        mask = np.zeros_like(img)
        history_img = [img_copy.copy()]
        history_mask = [mask.copy()]
        print("已清除当前掩码")

cv2.destroyAllWindows()