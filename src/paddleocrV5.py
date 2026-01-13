from paddleocr import PaddleOCR
from merge_verticle_text import merge_ocr_to_centers
from mark_points import draw_center_points_with_paths
from merge_vertical_1 import merge_with_line_scan

from sam2_utils import SAM2Segmenter
import cv2
import os
import numpy as np

img_path = "map_9.png"

# 初始化 PaddleOCR 实例
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False)

# 对示例图像执行 OCR 推理 
result = ocr.predict(
    input="./data/" + img_path)

# print(result[0])

# 可视化结果并保存 json 结果
# for res in result:
#     res.print()
#     res.save_to_img("output")
#     res.save_to_json("output")

# print(result[0]["rec_texts"])
# print(result[0]["rec_boxes"])

texts_merged, centers_merged = merge_ocr_to_centers(texts= result[0]["rec_texts"], boxes= result[0]["rec_boxes"])

print(texts_merged)
# print(centers_merged)

draw_center_points_with_paths(img_path, centers_merged)

#水平扫描合并
# texts_scan, centers_scan = merge_with_line_scan(
#     texts=result[0]["rec_texts"], 
#     boxes=result[0]["rec_boxes"], 
#     image_path=img_path
# )

# print("\n--- 水平扫描合并结果 ---")
# print(texts_scan)
# # print(centers_scan)

# # 2. 将结果绘制出来以便对比
# # 我们将结果保存为 "debug_scan_compare.png"
# draw_center_points_with_paths(
#     img_path, 
#     centers_scan
# )

# ==========================================
# 5. 调用 SAM2 进行分区上色
# ==========================================

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)

img_relative_path = "data/" + img_path
img_relative_path = os.path.join(root_dir, img_relative_path)
# 初始化 SAM2 (建议在循环外初始化一次，避免重复加载模型)
sam2_engine = SAM2Segmenter(model_type="tiny")

# 读取原图用于上色 (RGB 格式)
image_bgr = cv2.imread(img_relative_path)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# 创建一个用于上色的画布
color_overlay = image_bgr.copy()

print("正在通过 SAM2 生成分区...")

# 遍历每一个 OCR 中心点
for i, center in enumerate(centers_merged):
    # center 格式通常为 [x, y]
    point = [[center[0], center[1]]]
    
    # 获取该点对应的掩码
    # 注意：在 CPU 上，由于要对每张图提取特征，单次运行会稍慢
    mask = sam2_engine.get_mask_by_point(image_rgb, point)
    
    # 将 mask 转换为布尔型并确保维度正确
    mask_bool = mask.astype(bool)
    
    # 为每个分区分配随机颜色 (B, G, R)
    random_color = np.random.randint(0, 255, (3,)).tolist()
    
    # 对掩码区域进行上色
    color_overlay[mask_bool] = random_color
    
    print(f"已完成分区分割: {texts_merged[i]}")

# 6. 融合原图与上色层 (0.6 原图 + 0.4 颜色层)
final_output = cv2.addWeighted(image_bgr, 0.6, color_overlay, 0.4, 0)

# 7. 保存最终的分区结果
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

save_path = os.path.join(output_dir, "sam2_partition_" + img_path)
cv2.imwrite(save_path, final_output)

print(f"\n--- 任务完成 ---")
print(f"分区上色图已保存至: {save_path}")