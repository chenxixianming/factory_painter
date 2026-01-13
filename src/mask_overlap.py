import os
import cv2
import numpy as np
import re

class MaskOverlapFilter:
    def __init__(self, cache_dir='./cache', threshold=0.6):
        """
        初始化
        :param cache_dir: mask文件所在的文件夹路径
        :param threshold: 重叠阈值 (0.6 代表 60%)
        """
        self.cache_dir = cache_dir
        self.threshold = threshold
        self.masks = []
        self.filenames = []

    def _extract_number(self, filename):
        """
        从文件名中提取数字用于排序
        例如: 'mask_12.png' -> 12
        """
        numbers = re.findall(r'\d+', filename)
        return int(numbers[0]) if numbers else 0

    def load_masks(self):
        """
        读取并按编号排序 Mask
        """
        if not os.path.exists(self.cache_dir):
            raise FileNotFoundError(f"Directory not found: {self.cache_dir}")

        # 1. 获取所有图片文件
        valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif'}
        files = [f for f in os.listdir(self.cache_dir) 
                 if os.path.splitext(f)[1].lower() in valid_extensions]

        # 2. 按编号自然排序 (Human Sorting)
        # 确保 2.png 排在 10.png 前面
        files.sort(key=self._extract_number)
        self.filenames = files

        print(f"Found {len(files)} masks. Loading...")

        # 3. 读取图像
        loaded_masks = []
        for f in files:
            path = os.path.join(self.cache_dir, f)
            # 以灰度模式读取 (0)
            img = cv2.imread(path, 0)
            if img is None:
                print(f"Warning: Could not read {f}")
                continue
            
            # 二值化处理：将像素值 >0 的变为 1 (True)，其余为 0 (False)
            # 使用 uint8 节省内存，便于 OpenCV 计算
            binary_mask = (img > 0).astype(np.uint8)
            loaded_masks.append(binary_mask)
        
        self.masks = loaded_masks
        return self.masks

    def check_overlaps(self):
        """
        核心逻辑：检查重叠率
        :return: Boolean List
        """
        if not self.masks:
            self.load_masks()
        
        count = len(self.masks)
        # 初始化结果数组，默认全为 False
        result_flags = [False] * count
        
        # 为了加速计算，可以先计算每个 mask 自身的面积
        areas = [np.sum(m) for m in self.masks]

        # 双重循环比较
        for i in range(count):
            mask_i = self.masks[i]
            area_i = areas[i]

            # 如果自身面积为0（空mask），直接跳过
            if area_i == 0:
                result_flags[i] = False # 或者根据需求设为 True/False
                continue

            for j in range(count):
                if i == j:
                    continue # 不和自己比
                
                mask_j = self.masks[j]
                area_j = areas[j]

                # 优化点：利用 Bounding Box 预判断 (可选)
                # 如果两个物体的矩形框都不相交，像素肯定不相交，可以跳过这里以加速
                # 这里为了代码简洁，直接进行像素级计算
                
                # 1. 计算交集 (逻辑与)
                # cv2.bitwise_and 比 numpy.logical_and 在大图上通常更快
                intersection = cv2.bitwise_and(mask_i, mask_j)
                
                # 2. 计算交集面积
                # countNonZero 比 np.sum 快
                overlap_area = cv2.countNonZero(intersection)
                
                # 3. 判断条件：交集面积 > 自身面积 * 60%
                if overlap_area > (min(area_j, area_i) * self.threshold):
                    result_flags[i] = True
                    # 只要发现和一个重叠超过阈值，就可以停止内层循环，标记为 True
                    break 
        
        return result_flags

# --- 使用示例 ---
if __name__ == "__main__":
    # 假设你的根目录下有 /cache 文件夹
    checker = MaskOverlapFilter(cache_dir='./cache')
    
    # 运行
    try:
        flags = checker.check_overlaps()
        
        # 打印结果
        print("\nProcessing Result:")
        for name, is_overlapped in zip(checker.filenames, flags):
            status = "DELETE (Overlapped)" if is_overlapped else "KEEP"
            print(f"{name}: {status}")
            
        print(f"\nFinal Boolean Array: {flags}")
        
    except Exception as e:
        print(f"Error: {e}")