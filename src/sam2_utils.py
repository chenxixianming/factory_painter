import os
import sys
import torch
import cv2
import numpy as np

# 自动处理路径，确保能找到 SAM 文件夹
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sam_dir = os.path.join(root_dir, "SAM")
if sam_dir not in sys.path:
    sys.path.append(sam_dir)

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class SAM2Segmenter:
    def __init__(self, model_type="tiny"):
        """初始化模型，建议只执行一次"""
        self.device = torch.device("cpu")
        
        # 根据你的路径配置
        if model_type == "tiny":
            self.model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
            self.checkpoint = os.path.join(root_dir, "SAM/checkpoints/sam2.1_hiera_tiny.pt")
        
        print(f"正在加载 SAM 2 模型至 CPU，请稍候...")
        self.model = build_sam2(self.model_cfg, self.checkpoint, device=self.device)
        self.predictor = SAM2ImagePredictor(self.model)
        print("✅ SAM 2 模型加载成功")

    def get_mask_by_point(self, image_rgb, point_coords):
        """
        根据坐标点获取分割掩码
        :param image_rgb: RGB 格式的图像数组
        :param point_coords: [[x, y]] 坐标
        :return: mask (布尔矩阵)
        """
        # 1. 设置当前处理的图像（提取特征）
        self.predictor.set_image(image_rgb)
        
        # 2. 准备 Prompt
        input_point = np.array(point_coords)
        input_label = np.array([1] * len(input_point)) # 默认为前景点

        # 3. 执行预测
        masks, scores, _ = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        return masks
    
    def get_masks_by_boxes(self, image_rgb, boxes_list):
        """
        根据一组矩形框批量生成分割掩码 (Batch Inference)
        :param image_rgb: 图像数组 (RGB)
        :param boxes_list: 包含多个 Box 的列表，格式 [[x1, y1, x2, y2], ...]
        :return: masks (N, H, W) 的布尔矩阵，N 为 Box 的数量
        """
        # 1. 提取图像特征 (如果图片没变，这一步只会执行一次)
        self.predictor.set_image(image_rgb)
        
        # 2. 转换数据格式
        # SAM 2 要求输入为 (N, 4) 的 numpy 数组
        input_boxes = np.array(boxes_list)

        # 3. 批量预测
        # multimask_output=False 表示通过 Box 提示只返回一个最优的 Mask
        # 返回的 masks 形状通常是 (N, 1, H, W)
        masks, scores, _ = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False 
        )

        # 4. 降维处理
        # 去掉中间那个维度 1，变成 (N, H, W)，方便后续遍历使用
        if masks.ndim == 4:
            masks = masks.squeeze(1)
            
        return masks
    
# ==========================================
# 测试样例 (Test Case)
# ==========================================
if __name__ == "__main__":
    # 1. 实例化分割器
    segmenter = SAM2Segmenter(model_type="tiny")

    # 2. 模拟或读取一张测试图
    # 如果 data/map_9.png 不存在，会创建一个空白图进行逻辑测试
    test_img_path = os.path.join(root_dir, "data/map_9.png")
    
    if os.path.exists(test_img_path):
        image_bgr = cv2.imread(test_img_path)
    else:
        print("⚠️ 未找到测试图，正在创建模拟画布进行测试...")
        image_bgr = np.ones((600, 800, 3), dtype=np.uint8) * 255
        # 画一个矩形代表房间
        cv2.rectangle(image_bgr, (150, 150), (450, 450), (0, 0, 0), 2)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # 3. 设定测试点 (假设点在房间中心)
    test_point = [[300, 300]] 

    print(f"正在对坐标 {test_point} 进行分割测试...")
    
    # 4. 调用分割功能
    _, mask, _ = segmenter.get_mask_by_point(image_rgb, test_point)

    # 5. 可视化并保存结果
    # 将布尔型 Mask 转换为可视化图像
    # 我们创建一个半透明的红色层覆盖在原图上
    overlay = image_bgr.copy()
    # print(type(mask))
    mask_bool = mask.astype(bool)
    overlay[mask_bool] = [0, 0, 255] # 将掩码区域设为红色 (BGR)
    
    # 合成结果 (原图 70%，红色层 30%)
    output_img = cv2.addWeighted(image_bgr, 0.7, overlay, 0.3, 0)

    # 绘制测试点位
    cv2.circle(output_img, (test_point[0][0], test_point[0][1]), 5, (0, 255, 0), -1)

    # 创建输出文件夹并保存
    output_dir = os.path.join(root_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "sam2_test_render.png")
    
    cv2.imwrite(save_path, output_img)
    
    print(f"--- 测试完成 ---")
    print(f"1. 掩码形状: {mask.shape}")
    print(f"2. 掩码覆盖像素点数: {np.sum(mask)}")
    print(f"3. 可视化结果已保存至: {save_path}")