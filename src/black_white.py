import cv2
import numpy as np
import os
import glob

def process_by_saturation_filter():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    data_dir = os.path.join(root_dir, "data")

    image_files = []
    for ext in ('*.png', '*.jpg', '*.jpeg'):
        image_files.extend(glob.glob(os.path.join(data_dir, ext)))
    
    for img_path in sorted(image_files):
        filename = os.path.basename(img_path)
        if filename.startswith("bw_"): continue # 跳过已处理文件
        
        img = cv2.imread(img_path)
        if img is None: continue
        
        # 1. 转到 HSV 空间
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 2. 定义“无彩色”（黑白灰）的过滤范围
        # H (色调): 0-180 (全部覆盖)
        # S (饱和度): 0 - 50 (关键！数值越低，过滤色彩越狠。50以下基本就是灰度色)
        # V (亮度): 0 - 200 (放宽到200，确保高亮度的灰线也能保留)
        lower_gray = np.array([0, 0, 0])
        upper_gray = np.array([180, 110, 240]) 
        
        # 3. 创建掩码：只保留低饱和度的像素
        gray_mask = cv2.inRange(hsv, lower_gray, upper_gray)
        
        # 4. 建立纯白画布
        result = np.ones_like(img) * 255
        
        # 5. 将符合“无彩色”条件的像素恢复（支持保留灰线原始色调）
        result[gray_mask > 0] = img[gray_mask > 0]

        # 保存结果
        save_path = os.path.join(data_dir, f"bw_{filename}")
        cv2.imwrite(save_path, result)
        print(f"处理完成（饱和度过滤模式）: {filename}")

if __name__ == "__main__":
    process_by_saturation_filter()