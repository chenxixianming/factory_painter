import cv2
import numpy as np
import os

class LineExtender:
    def __init__(self, input_dir='./cache/no_char', output_dir='./cache/111'):
        self.input_dir = input_dir
        self.output_dir = output_dir

    def skeletonize(self, img):
        """
        OpenCV 标准骨架化算法 (无需 ximgproc 库)
        """
        size = np.size(img)
        skel = np.zeros(img.shape, np.uint8)
        
        # 这里的 element 决定了骨架的连通性
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        done = False
        
        temp_img = img.copy()
        
        while not done:
            eroded = cv2.erode(temp_img, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(temp_img, temp)
            skel = cv2.bitwise_or(skel, temp)
            temp_img = eroded.copy()
            
            zeros = size - cv2.countNonZero(temp_img)
            if zeros == size:
                done = True
        return skel

    def extend_lines(self, img, gap_size=5, target_thickness=1, save = False):
        """
        延长线条并保持宽度
        :param img: 二值图 (黑底白线)
        :param gap_size: 能连接的最大断点距离 (相当于延长的长度)
        :param target_thickness: 最终输出的线条宽度 (半径，1代表3px宽)
        """
        # 1. 重度膨胀：连接断点，但会变粗
        # kernel 大小决定了能连接多远的虚线
        connect_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gap_size, gap_size))
        fat_lines = cv2.dilate(img, connect_kernel, iterations=1)
        
        # 2. 骨架化：把变粗的线条削减为 1px 宽
        # 这一步是核心，它把"宽度"丢掉了，但保留了"长度"和"连接性"
        skeleton = self.skeletonize(fat_lines)
        
        # 3. 恢复宽度：按照需求重新赋予宽度
        if target_thickness > 0:
            restore_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            final_img = cv2.dilate(skeleton, restore_kernel, iterations=target_thickness)
        else:
            final_img = skeleton
        
        # if save:
            # output_path = os.path.join(self.output_dir, img_name)
            # cv2.imwrite(output_path, img)
            # print(f"✨ Cleaned image saved to: {output_path}")
            


        return final_img

# --- 使用示例 ---
# 假设 img 是你的二值化图
# extender = LineExtender()
# result = extender.extend_lines(img, gap_size=7, target_thickness=1)

if __name__ == "__main__":
    input_dir = './cache/clean'
    output_dir = './cache/111'  # 确保这里定义了
    os.makedirs(output_dir, exist_ok=True) # 确保输出文件夹存在

    extender = LineExtender()

    if os.path.exists(input_dir):
        for f in os.listdir(input_dir):
            if f.lower().endswith(('.png', '.jpg')):
                full_path = os.path.join(input_dir, f)
                
                # 1. 读取图片 (灰度)
                img = cv2.imread(full_path, 0)
                
                if img is None:
                    print(f"Error: Cannot read {full_path}")
                    continue
                
                # ==========================================
                # 【关键修复步骤】
                # 检查图片是否为白底黑线，如果是，则反转为黑底白线
                # 简单的判断方法：如果大部分像素是白的(平均值>127)，就反转
                # ==========================================
                if np.mean(img) > 127:
                    print(f"Detected light background for {f}, inverting...")
                    img_input = cv2.bitwise_not(img)
                else:
                    img_input = img

                print(f"Processing {f}...")
                
                # 2. 传入黑底白线的图片进行处理
                # 注意：gap_size 不要设太大，否则会把平行的两堵墙连在一起
                result_skeleton = extender.extend_lines(img_input, gap_size=5, target_thickness=1)
                
                # 3. (可选) 如果你想保存为原来的“白底黑线”风格，需要再次反转回来
                final_result = cv2.bitwise_not(result_skeleton)
                
                # 4. 保存结果
                save_path = os.path.join(output_dir, f"extended_{f}")
                cv2.imwrite(save_path, final_result)
                print(f"Saved to {save_path}")