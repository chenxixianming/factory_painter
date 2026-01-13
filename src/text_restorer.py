import os
import cv2
import numpy as np

class TextRestorer:
    def __init__(self, data_dir='./data', output_dir='./output', cache_dir='./cache'):
        """
        初始化路径配置
        :param data_dir: 原图存放目录
        :param output_dir: 处理后图片及最终结果存放目录
        :param cache_dir: 缓存目录 (存放 mask)
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.cache_dir = cache_dir

    def run(self, img_name):
        """
        执行文字回填操作
        :param img_name: 图片文件名 (例如 'factory.jpg')
        :return: 保存的新图片路径
        """
        # --- 1. 构建文件路径 ---
        # 原图路径: ./data/{img_name}
        origin_path = os.path.join(self.data_dir, img_name)
        
        # 处理后的图路径: ./output/processed_{img_name}
        processed_path = os.path.join(self.output_dir, f"processed_{img_name}")
        
        # Mask 路径: ./cache/character_mask/character_mask.png (固定路径)
        mask_path = os.path.join(self.cache_dir, "character_mask", "character_mask.png")

        # --- 2. 读取图片 ---
        print(f"Loading images for restoration...")
        
        origin_img = cv2.imread(origin_path)
        if origin_img is None:
            raise FileNotFoundError(f"Original image not found: {origin_path}")

        processed_img = cv2.imread(processed_path)
        if processed_img is None:
            raise FileNotFoundError(f"Processed image not found: {processed_path}")

        # Mask 读取 (保持RGB模式，以便判断白色)
        mask_img = cv2.imread(mask_path)
        
        # 如果 mask 不存在，直接返回处理后的图，或者报错
        if mask_img is None:
            print(f"Warning: Mask not found at {mask_path}. Skipping restoration.")
            return processed_path

        # --- 3. 尺寸校验与对齐 ---
        # 以原图尺寸为基准
        h, w = origin_img.shape[:2]
        
        # 确保 processed_img 尺寸一致
        if processed_img.shape[:2] != (h, w):
            processed_img = cv2.resize(processed_img, (w, h))
        
        # 确保 mask_img 尺寸一致
        if mask_img.shape[:2] != (h, w):
            mask_img = cv2.resize(mask_img, (w, h), interpolation=cv2.INTER_NEAREST)

        # --- 4. 核心逻辑：根据 Mask 回填像素 ---
        # 目标：创建一个 boolean mask，其中 Mask 图中为纯白色的地方为 True
        # 定义白色的范围 (BGR: 255, 255, 255)
        lower_white = np.array([255, 255, 255], dtype=np.uint8)
        upper_white = np.array([255, 255, 255], dtype=np.uint8)
        
        # 生成二值掩膜 (white_region 中，白色区域值为 255，其余为 0)
        white_region = cv2.inRange(mask_img, lower_white, upper_white)
        
        # 创建画布，以 processed_img 为底
        final_img = processed_img.copy()
        
        # 关键步骤：利用 numpy 布尔索引进行像素替换
        # 将 final_img 中对应 Mask 为白色的位置，替换为 origin_img 中对应位置的像素
        final_img[white_region > 0] = origin_img[white_region > 0]

        # --- 5. 保存结果 ---
        # 新文件名: write_back_{img_name}
        save_name = f"write_back_{img_name}"
        save_path = os.path.join(self.output_dir, save_name)
        
        cv2.imwrite(save_path, final_img)
        print(f"✅ Text restored and saved to: {save_path}")
        
        return save_path

# --- 测试代码 ---
if __name__ == "__main__":
    # 假设你在项目根目录下运行
    restorer = TextRestorer()
    try:
        # 请确保 ./data/factory.jpg 和 ./output/processed_factory.jpg 存在
        restorer.run("map_10.png")
    except Exception as e:
        print(f"Error: {e}")