import os
import sys
import torch
import cv2
import numpy as np

# 1. 设置路径逻辑 (确保能够找到 SAM 文件夹)
current_dir = os.path.dirname(os.path.abspath(__file__)) # src
root_dir = os.path.dirname(current_dir)                 # factory_painter
sam_dir = os.path.join(root_dir, "SAM")                 # /workspaces/factory_painter/SAM

# 将 SAM 路径加入 sys.path
if sam_dir not in sys.path:
    sys.path.append(sam_dir)

# 2. 导入 SAM 2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# 3. 强制 CPU 模式
device = torch.device("cpu")

# 4. 指定路径 (修复 MissingConfigException 的关键)
# 注意：sam2_hiera_t.yaml 必须在 SAM/sam2/configs/ 目录下存在
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml" 
sam2_checkpoint = os.path.join(root_dir, "SAM/checkpoints/sam2.1_hiera_tiny.pt")

# 5. 初始化模型
# build_sam2 内部会从 Python 搜索路径下的 'sam2' 包目录寻找配置
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2_model)

print("✅ SAM 2 模型及配置加载成功！")

# --- 4. 读取图像 ---
image_path = "data/map_9.png"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(image)

# --- 5. 设置提示 (Prompt) ---
# 假设你想分割坐标为 [x=200, y=300] 的区域
input_point = np.array([[200, 300]])
input_label = np.array([1]) # 1 代表前景点

# --- 6. 执行预测 ---
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=False,
)

# --- 7. 保存结果 ---
# masks[0] 是一个布尔矩阵，将其转换为图片保存
mask_img = (masks[0] * 255).astype(np.uint8)
cv2.imwrite("output/sam2_mask.png", mask_img)
print("分割完成，结果已保存至 output/sam2_mask.png")