import cv2
import os
import numpy as np
import json
from ocr_utils import IndustrialOCRManager
from sam2_utils import SAM2Segmenter

# --- 1. 配置路径 ---
img_name = "map_8.png"
img_path = os.path.join("data", img_name)
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# --- 2. 初始化两大引擎 (只做一次) ---
ocr_manager = IndustrialOCRManager()
# sam2_engine = SAM2Segmenter(model_type="tiny")

# --- 3. 一键获取 OCR 合并结果 ---
texts, centers, boxes = ocr_manager.get_merged_results(img_path)

ocr_data = []
for i in range(len(texts)):
    ocr_data.append({
        "id": i,
        "text": texts[i],
        "center": [float(centers[i][0]), float(centers[i][1])], # 确保是 float
        "box": [float(boxes[i][0]), float(boxes[i][1]), float(boxes[i][2]), float(boxes[i][3])]
    })

# 保存为 JSON 文件
json_path = os.path.join(output_dir, f"{img_name}.json")
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(ocr_data, f, ensure_ascii=False, indent=4)

print(f"✅ OCR 结果已保存至: {json_path}")


json_path = os.path.join("output", f"{img_name}.json")
with open(json_path, 'r', encoding='utf-8') as f:
    ocr_results = json.load(f)

boxes = [item["box"] for item in ocr_results]

if not boxes:
    print("JSON 中没有找到 Box 数据")
    exit()

# 2. 初始化 SAM 2
sam2 = SAM2Segmenter(model_type="tiny")



#-----------------------------------------------
#去除文字
#-----------------------------------------------

# 3. 读取图像
image_bgr = cv2.imread(img_path)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
color_overlay = image_bgr.copy()

# 4. 【核心调用】批量获取所有 Masks
# 这一步非常快，因为是一次性推理
print(f"正在为 {len(boxes)} 个区域生成掩码...")
all_masks = sam2.get_masks_by_boxes(image_rgb, boxes)

# 5. 遍历 Mask 进行上色
for i, mask in enumerate(all_masks):
    # mask 已经是 (H, W) 的布尔或 0/1 矩阵
    mask_bool = mask.astype(bool)
    
    # 随机颜色
    # color = np.random.randint(0, 255, (3,)).tolist()
    color = [255, 255, 255]
    
    # 上色
    color_overlay[mask_bool] = color
    
    # 打印对应的文字（方便调试）
    print(f"已处理: {ocr_data[i]['text']}")

# 6. 保存结果
result = cv2.addWeighted(image_bgr, 0.7, color_overlay, 0.3, 0)
save_path = os.path.join("output", f"sam2_box_whight_colored_{img_name}")
# cv2.imwrite(save_path, result)
cv2.imwrite(save_path, color_overlay)
print(f"✅ 结果已保存至: {save_path}")




#-----------------------------------------------
#SAM在去除文字的图上分区
#-----------------------------------------------

centers = [item["center"] for item in ocr_results]

# # --- 4. SAM2 分区上色 ---
sam2_engine = SAM2Segmenter(model_type="tiny")
image_bgr = color_overlay
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
color_overlay = image_bgr.copy()

all_masks = sam2.get_mask_by_point(image_rgb, centers)

sam2_engine.predictor.set_image(image_rgb)

# --- 2. 遍历 JSON 数据进行上色 ---
for item in ocr_results:
    text = item["text"]
    center = item["center"] # [x, y]
    
    # 解决“只涂文字”的关键点：使用多掩码输出并选择最大范围
    masks, scores, _ = sam2_engine.predictor.predict(
        point_coords=np.array([center]),
        point_labels=np.array([1]),
        multimask_output=True # 必须为 True
    )
    
    # 选择得分最高或范围最广的 mask (通常 masks[2] 范围最大)
    # 你可以根据实际效果尝试 masks[np.argmax(scores)] 或固定 masks[1]
    best_mask = masks[np.argmax(scores)].astype(bool)
    
    # 随机颜色并上色
    color = np.random.randint(0, 255, (3,)).tolist()
    color_overlay[best_mask] = color
    print(f"已从 JSON 读取并上色: {text}")

# --- 3. 合成保存 ---
final_res = cv2.addWeighted(image_bgr, 0.7, color_overlay, 0.3, 0)
cv2.imwrite(os.path.join("output", f"point_colored_{img_name}"), final_res)

# for i, mask in enumerate(all_masks):
#     # mask 已经是 (H, W) 的布尔或 0/1 矩阵
#     mask_bool = mask.astype(bool)
    
#     # 随机颜色
#     color = np.random.randint(0, 255, (3,)).tolist()
#     # color = [255, 255, 255]
    
#     # 上色
#     color_overlay[mask_bool] = color
    
#     # 打印对应的文字（方便调试）
#     print(f"已处理: {ocr_data[i]['text']}")

# # 6. 保存结果
# result = cv2.addWeighted(image_bgr, 0.7, color_overlay, 0.3, 0)
# save_path = os.path.join("output", f"sam2_point_colored_{img_name}")
# # cv2.imwrite(save_path, result)
# cv2.imwrite(save_path, result)
# print(f"✅ 结果已保存至: {save_path}")

# print("开始分区上色...")
# for item in ocr_results:
#     text = item["text"]
#     center = item["center"] # [x, y]
#     box = item["box"]
    
#     # 解决“只涂文字”的关键点：使用多掩码输出并选择最大范围
#     masks, scores, _ = sam2_engine.predictor.predict(
#         point_coords=np.array([center]),
#         point_labels=np.array([2]),
#         box = 
#         multimask_output=False # 必须为 True
#     )
    
#     # 选择得分最高或范围最广的 mask (通常 masks[2] 范围最大)
#     # 你可以根据实际效果尝试 masks[np.argmax(scores)] 或固定 masks[1]
#     # best_mask = masks[np.argmax(scores)].astype(bool)
#     best_mask = masks[0].astype(bool)
    
#     # 随机颜色并上色
#     # color = np.random.randint(0, 255, (3,)).tolist()
#     color = [0, 255, 255]
#     color_overlay[best_mask] = color
#     print(f"已从 JSON 读取并上色: {text}")

# # --- 5. 结果合成与保存 ---
# # final_res = cv2.addWeighted(image_bgr, 0.6, color_overlay, 0.4, 0)
# save_path = os.path.join(output_dir, f"final_result_{img_name}")
# # cv2.imwrite(save_path, final_res)
# cv2.imwrite(save_path, color_overlay)

# print(f"🎉 全部处理完成！保存至: {save_path}")