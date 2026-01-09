import cv2
import os

# --- 1. 获取路径的通用逻辑 ---
# 获取当前脚本(draw_tool.py)的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 向上跳一级，到达项目根目录 (factory_painter)
project_root = os.path.dirname(current_dir)

def draw_center_points_with_paths(image_name, center_coords):
    """
    :param image_name: 仅传入图片文件名，如 'map.png'
    """

    pure_file_name = os.path.basename(image_name)
    
    input_path = os.path.join(project_root, "data", pure_file_name)
    output_dir = os.path.join(project_root, "output")
    output_path = os.path.join(output_dir, f"debug_{pure_file_name}")
    
    # 拼接输入和输出的完整路径
    input_path = os.path.join(project_root, "data", image_name)
    output_dir = os.path.join(project_root, "output")
    output_path = os.path.join(output_dir, f"debug_{image_name}")

    # 检查输出目录是否存在，不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- 2. OpenCV 处理逻辑 ---
    img = cv2.imread(input_path)
    if img is None:
        print(f"错误：无法找到图片 {input_path}")
        return

    for x, y in center_coords:
        # 绘制 5x5 红点
        cv2.rectangle(img, (int(x)-2, int(y)-2), (int(x)+2, int(y)+2), (0, 0, 255), -1)

    # 保存
    cv2.imwrite(output_path, img)
    print(f"标注完成！\n输入位置: {input_path}\n输出位置: {output_path}")

# 调用示例
if __name__ == "__main__":
    draw_center_points_with_paths("lQDPKdU6o7Q9-UPNAfTNBIKwsI-AMOPGxKwICjTC7zs2AA_1154_500.jpg", [[150, 200]])