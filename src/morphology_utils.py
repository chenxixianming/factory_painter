import cv2
import numpy as np
import os
import glob

class MorphologyProcessor:
    def __init__(self, kernel_size=5, shape="rect"):
        """
        初始化形态学处理器
        :param kernel_size: 核大小，数值越大，连接缝隙的能力越强。必须是奇数 (3, 5, 7, 9...)
                            - 3: 填补极小缝隙，轻微加粗
                            - 5-7: 填补普通虚线断点 (推荐起始值)
                            - 9以上: 填补较大缺口，线条会变得很粗
        :param shape: 核形状，'rect' (矩形，连接能力最强) 或 'ellipse' (椭圆，转角更平滑)
        """
        self.kernel_size = kernel_size
        # 定义结构元素 (Kernel)
        if shape == "rect":
            self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        elif shape == "ellipse":
            self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        else:
            self.kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        print(f"✨ 形态学处理器初始化完成 (Mode: Erosion for Black Lines, Kernel: {kernel_size})")

    def _get_paths(self, filename):
        """内部辅助方法：构建路径"""
        current_dir = os.path.dirname(os.path.abspath(__file__)) # src/
        root_dir = os.path.dirname(current_dir)                 # 项目根目录
        output_dir = os.path.join(root_dir, "output")           # output/
        input_path = os.path.join(output_dir, filename)
        return input_path, output_dir

    # ==========================================
    # 核心方法：腐蚀 (用于加粗黑线/连接虚线)
    # ==========================================
    def erode_file(self, filename, output_suffix="_connected"):
        """
        【专门处理黑线】读取图片，执行腐蚀操作，让黑色区域扩张，连接断点。
        """
        input_path, output_dir = self._get_paths(filename)
        
        if not os.path.exists(input_path):
            print(f"❌ 错误：在 output 文件夹中找不到文件: {filename}")
            return None

        # 读取图像 (保持原色读入即可)
        img = cv2.imread(input_path)
        if img is None:
            print(f"❌ 错误：无法解码图像: {filename}")
            return None

        print(f"正在处理: {filename}，尝试连接黑色线条...")

        # --- 核心操作：腐蚀 (Erosion) ---
        # 原理：对于核覆盖的区域，取像素最小值。
        # 在白底(255)黑线(0)图中，只要核范围内有一个黑点(0)，中心就会变成黑点。
        connected_img = cv2.erode(img, self.kernel, iterations=1)

        # 保存结果
        name, ext = os.path.splitext(filename)
        save_name = f"{name}{output_suffix}{ext}"
        save_path = os.path.join(output_dir, save_name)
        
        cv2.imwrite(save_path, connected_img)
        print(f"✅ 已保存连接后的图像: {save_name}")
        
        return connected_img

    # 保留膨胀方法供参考 (用于加粗白色区域)
    def dilate_file(self, filename, output_suffix="_dilated"):
        input_path, output_dir = self._get_paths(filename)
        if not os.path.exists(input_path): return None
        img = cv2.imread(input_path)
        # 膨胀：让亮色区域扩张
        dilated_img = cv2.dilate(img, self.kernel, iterations=1)
        name, ext = os.path.splitext(filename)
        save_path = os.path.join(output_dir, f"{name}{output_suffix}{ext}")
        cv2.imwrite(save_path, dilated_img)
        print(f"✅ 膨胀完成 (白色区域变大): {save_path}")
        return dilated_img

# ==========================================
# 测试与使用示例
# ==========================================
if __name__ == "__main__":
    # 1. 设置参数
    # kernel_size 是关键！
    # 如果虚线间隔比较大，尝试设置成 7 或 9。如果间隔小，用 5。
    KERNEL_SIZE = 9
    
    target_filename = "sam2_box_whight_colored_map_6.jpg" # 请确保 output 文件夹里有这个文件

    # 2. 初始化处理器
    # 使用 'rect' 矩形核，连接直线断点的能力最强
    processor = MorphologyProcessor(kernel_size=KERNEL_SIZE, shape="rect")

    # 3. 执行黑线连接操作 (使用 erode_file)
    print(f"\n--- 开始处理 ---")
    processor.erode_file(target_filename, output_suffix="_connected_k" + str(KERNEL_SIZE))
    print(f"--- 处理结束，请检查 output 文件夹 ---")