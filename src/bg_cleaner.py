import os
import cv2
import numpy as np

class BackgroundCleaner:
    def __init__(self, input_dir='./cache/no_char', output_dir='./cache/clean', threshold=200):
        """
        初始化背景净化器
        :param input_dir: 输入文件夹
        :param output_dir: 输出文件夹
        :param threshold: 净化阈值 (0-255)。
                          值越大，清理力度越小（保留更多细节，包括噪点）；
                          值越小，清理力度越大（噪点消失，但浅色线条也可能消失）。
                          推荐范围: 180 - 230。
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.threshold = threshold
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)

    def process(self, img_name, save = False):
        """
        读取图片，净化背景，并保存
        :param img_name: 文件名 (例如 '1.png')
        :return: 处理后的文件路径
        """
        input_path = os.path.join(self.input_dir, img_name)
        output_path = os.path.join(self.output_dir, img_name)
        
        # 1. 读取图片
        img = cv2.imread(input_path)
        if img is None:
            print(f"⚠️ Warning: Could not read image at {input_path}")
            return None
            
        # 2. 转换为灰度图 (用于计算亮度)
        # 我们基于灰度值来判断哪个像素是噪点
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 3. 核心逻辑：条件过滤
        # 逻辑：如果一个像素的灰度值 > threshold (代表它是浅色/白色的)，
        #      则将其 BGR 三通道全部设为 255 (纯白)。
        #      否则 (深色/墙线)，保持原样。
        
        # 创建一个布尔掩码：找到所有“不够黑”的像素
        # gray > self.threshold 会返回一个 True/False 的矩阵
        mask = gray > self.threshold
        
        # 将原图中对应的像素设为纯白
        img[mask] = [255, 255, 255]
        
        # (可选) 增强对比度：让留下的线条更黑
        # 如果觉得线条太浅，可以取消下面这行的注释
        img[~mask] = (img[~mask] * 0).astype(np.uint8) 

        # ==========================================
        # [新增逻辑] Step 4: 去除孤立噪点 (基于密度)
        # 要求：检查5*5范围内黑色像素点小于5，则删除
        # ==========================================
        
        # 4.1 重新计算灰度图（因为图像已经被修改过了）
        gray_cleaned = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 4.2 生成二值掩码：定义什么是“黑色像素点”
        # 在这一步，凡是不是纯白(255)的，都算作黑色像素点。
        # 我们用 1 代表黑色像素，0 代表白色背景，方便后面计数运算。
        # astype(np.uint8) 会把 True/False 变成 1/0。
        dark_pixel_mask = (gray_cleaned < 255).astype(np.uint8)

        # 4.3 定义 5x5 的卷积核 (全1矩阵)
        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # 4.4 核心操作：使用 filter2D 计算邻域密度
        # 这步操作后，neighbor_count_img 中每个像素的值，
        # 就是原mask中以它为中心的 5x5 区域内黑色像素的总数。
        # ddepth=-1 表示输出图像深度与原图相同 (uint8)
        neighbor_count_img = cv2.filter2D(dark_pixel_mask, -1, kernel)

        # 4.5 找到符合条件的噪点
        # 条件 A: 它本身是个黑色像素点 (dark_pixel_mask == 1)
        # 条件 B: 它周围 5x5 区域内的黑色像素总数小于 5 (neighbor_count_img < 5)
        noise_threshold = 4
        noise_to_remove_mask = (dark_pixel_mask == 1) & (neighbor_count_img < noise_threshold)
        
        # 统计一下要删除多少个噪点像素 (调试用)
        noise_count = np.sum(noise_to_remove_mask)
        if noise_count > 0:
             print(f"   -> Found and removed {noise_count} isolated noise pixels.")

        # 4.6 将这些噪点变白
        # 使用布尔索引直接修改原图 BGR 像素
        img[noise_to_remove_mask] = [255, 255, 255]

        # ==========================================
        # Step 5: 保存图片
        # ==========================================
        if save:
            cv2.imwrite(output_path, img)
            print(f"✨ Cleaned image saved to: {output_path}")
            
            return output_path
        
        return img

# --- 使用示例 ---
if __name__ == "__main__":
    # 实例化
    # 建议先用 210 试一下。如果噪点没去干净，就把数字改小 (比如 190)；
    # 如果墙线断了，就把数字改大 (比如 230)。
    cleaner = BackgroundCleaner(threshold=210)
    
    # 假设你的 ./cache/no_char 文件夹里有一张名为 test.png 的图
    # 你可以遍历文件夹运行
    if os.path.exists('./cache/no_char'):
        for f in os.listdir('./cache/no_char'):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                cleaner.process(f, save= True)