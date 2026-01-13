import os
import cv2
import numpy as np
import re

class ResultVisualizer:
    def __init__(self, cache_dir='./cache', output_dir='./output'):
        """
        初始化可视化器
        :param cache_dir: mask存放的目录，用于读取需要标绿的mask
        :param output_dir: 最终图像保存的目录
        """
        self.cache_dir = cache_dir
        self.output_dir = output_dir
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 预先加载并排序 mask 文件路径，确保索引对应准确
        self.sorted_mask_paths = self._get_sorted_mask_paths()

    def _extract_number(self, filename):
        """辅助函数：用于文件名数字排序"""
        numbers = re.findall(r'\d+', filename)
        return int(numbers[0]) if numbers else 0

    def _get_sorted_mask_paths(self):
        """获取排序后的所有 mask 文件完整路径"""
        if not os.path.exists(self.cache_dir):
            print(f"Warning: Cache directory {self.cache_dir} does not exist.")
            return []
            
        valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif'}
        files = [f for f in os.listdir(self.cache_dir) 
                 if os.path.splitext(f)[1].lower() in valid_extensions]
        # 关键：使用与上一个任务相同的排序逻辑
        files.sort(key=self._extract_number)
        return [os.path.join(self.cache_dir, f) for f in files]

    def _expand_box(self, box, img_shape, scale=1.5):
        """
        辅助函数：将边界框以中心为基点扩大指定倍数，并限制在图像范围内
        :param box: [x1, y1, x2, y2] (左上角x, 左上角y, 右下角x, 右下角y)
        :param img_shape: (height, width, channels) 用于边界截断
        :param scale: 缩放倍数
        """
        x1, y1, x2, y2 = box
        h_img, w_img = img_shape[:2]
        
        # 计算中心点和原始宽高
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        
        # 计算新宽高
        new_w, new_h = w * scale, h * scale
        
        # 计算新坐标
        nx1 = int(cx - new_w / 2)
        ny1 = int(cy - new_h / 2)
        nx2 = int(cx + new_w / 2)
        ny2 = int(cy + new_h / 2)
        
        # 截断坐标，防止超出图像边界 (Clipping)
        nx1 = max(0, nx1)
        ny1 = max(0, ny1)
        nx2 = min(w_img, nx2)
        ny2 = min(h_img, ny2)
        
        return nx1, ny1, nx2, ny2

    def draw_and_save(self, image_path, boxes, result_flags, output_filename='visualized_result.png'):
        """
        执行绘制并保存
        :param image_path: 原始图片路径
        :param boxes: 边界框列表，格式 [[x1,y1,x2,y2], ...]
        :param result_flags: 布尔列表 [True, False, ...]
        :param output_filename: 保存的文件名
        """
        # 1. 读取原始图像
        base_img = cv2.imread(image_path)
        if base_img is None:
            raise FileNotFoundError(f"Could not load source image: {image_path}")
            
        # 复制一份图像用于绘制，避免修改原图数据（虽然这里读进来已经是副本了，是个好习惯）
        canvas_img = base_img.copy()
        img_shape = canvas_img.shape
        
        # 定义颜色 (BGR 格式)
        COLOR_YELLOW = (0, 255, 255)
        COLOR_GREEN = (0, 255, 0)

        # 校验数据长度一致性
        num_items = len(result_flags)
        if len(boxes) != num_items:
             raise ValueError("Length of boxes and result_flags must match.")
        if len(self.sorted_mask_paths) < num_items:
             print("Warning: Not enough mask files found in cache dir for False flags.")

        print(f"Processing {num_items} items...")

        # 2. 循环处理每一项
        for n in range(num_items):
            is_overlapped = result_flags[n]
            
            if is_overlapped:
                # --- 情况 True: 绘制扩大的黄色方框 ---
                current_box = boxes[n]
                # 计算扩大后的坐标
                nx1, ny1, nx2, ny2 = self._expand_box(current_box, img_shape, scale=1.5)
                # 绘制实心矩形 (thickness=-1 表示填充)
                cv2.rectangle(canvas_img, (nx1, ny1), (nx2, ny2), COLOR_YELLOW, thickness=-1)
                
            else:
                # --- 情况 False: 绘制绿色 Mask ---
                # 确保有对应的 mask 文件
                if n < len(self.sorted_mask_paths):
                    mask_path = self.sorted_mask_paths[n]
                    # 以灰度模式读取 mask
                    mask = cv2.imread(mask_path, 0)
                    
                    if mask is not None:
                        # 确保 mask 大小和原图一致 (有时候 mask 可能会有细微尺寸差异导致报错)
                        if mask.shape != img_shape[:2]:
                             mask = cv2.resize(mask, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_NEAREST)

                        # 核心操作：找到 mask 中大于 0 的区域，将画布对应位置设为绿色
                        # canvas_img[mask > 0] 选择的是符合条件的像素点索引
                        canvas_img[mask > 0] = COLOR_GREEN
                    else:
                        print(f"Error reading mask: {mask_path}")

        # 3. 保存结果图像
        output_path = os.path.join(self.output_dir, output_filename)
        cv2.imwrite(output_path, canvas_img)
        print(f"Result saved to: {output_path}")
        return output_path

# --- 使用示例 ---
# (通常这个类会和上一个类配合使用)
if __name__ == "__main__":
    # 1. 准备模拟数据 (假设这些数据来自上一轮处理和你的目标检测模型)
    
    # 假设原始图片是一个 500x500 的黑色背景图，我们先创建一个用于测试
    test_img_path = "test_source_image.png"
    dummy_img = np.zeros((500, 500, 3), dtype=np.uint8)
    cv2.imwrite(test_img_path, dummy_img)

    # 模拟传入的数据
    # 假设有3个物体
    # 物体0: 被标记为重叠(True)，需要画黄色大框
    # 物体1: 没重叠(False)，需要读取 cache/2.png 画绿色mask (注意文件名编号是自然排序的第2个)
    # 物体2: 被标记为重叠(True)，测试边界截断
    
    mock_boxes = [
        [100, 100, 200, 200], # Box 0 (中心 150,150，宽高100 -> 新宽高150)
        [300, 300, 350, 350], # Box 1 (这个数据其实这里用不到，因为False是读mask)
        [450, 450, 480, 480]  # Box 2 (靠近边界，测试截断)
    ]
    # 假设上一个任务返回的 flags 是 [True, False, True]
    # 这意味着我们需要读取 cache 文件夹里排序后的第2张 mask 图片 (索引为1)
    mock_flags = [True, False, True] 
    
    # 确保cache文件夹和里面有测试mask存在 (为了运行示例需要手动创建)
    os.makedirs('./cache', exist_ok=True)
    # 创建一个模拟的 2.png mask
    dummy_mask = np.zeros((500, 500), dtype=np.uint8)
    cv2.rectangle(dummy_mask, (300, 300), (350, 350), 255, -1) # 对应物体1的位置
    cv2.imwrite('./cache/mask_2.png', dummy_mask)
    # 创建其他干扰mask确保排序正确
    cv2.imwrite('./cache/mask_1.png', np.zeros((500,500), dtype=np.uint8))
    cv2.imwrite('./cache/mask_3.png', np.zeros((500,500), dtype=np.uint8))


    # 2. 初始化可视化器
    visualizer = ResultVisualizer(cache_dir='./cache', output_dir='./output')
    
    # 3. 执行绘制
    try:
        visualizer.draw_and_save(
            image_path=test_img_path,
            boxes=mock_boxes,
            result_flags=mock_flags,
            output_filename='final_visualization.png'
        )
        print("\nVisualization complete. Check the output folder.")
        
        # 清理测试产生的临时文件 (可选)
        # os.remove(test_img_path)
        # import shutil
        # shutil.rmtree('./cache')

    except Exception as e:
        print(f"An error occurred: {e}")