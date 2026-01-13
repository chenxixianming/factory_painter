import cv2
import os
import numpy as np
import json
from ocr_utils import IndustrialOCRManager
from sam2_utils import SAM2Segmenter
from mask_overlap import MaskOverlapFilter
from result_drawer import ResultVisualizer
from text_restorer import TextRestorer


# --- 1. 配置路径 ---
img_name = "map_8.png"
img_path = os.path.join("data", img_name)
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)



#------------------------------------------------
#OCR开始
#------------------------------------------------



# --- 2. 初始化两大引擎 (只做一次) ---
ocr_manager = IndustrialOCRManager()
# sam2_engine = SAM2Segmenter(model_type="tiny")

# --- 3. 一键获取 OCR 合并结果 ---
texts, centers, boxes = ocr_manager.get_ocr_results(img_path, verticle_merge= True)

# texts, boxes = ocr_manager.get_ocr_results(img_path)
# texts, centers, boxes = ocr_manager.correct_structure_with_llm(texts, boxes)

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




#------------------------------------------------------------------
#OCR结束，SAM开始
#------------------------------------------------------------------






json_path = os.path.join("output", f"{img_name}.json")
with open(json_path, 'r', encoding='utf-8') as f:
    ocr_results = json.load(f)

boxes = [item["box"] for item in ocr_results]

boxes_copy = boxes

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
character_mask = np.zeros_like(image_bgr)

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
    character_mask[mask_bool] = color
    
    # 打印对应的文字（方便调试）
    print(f"已处理: {ocr_results[i]['text']}")

# 6. 保存结果
# result = cv2.addWeighted(image_bgr, 0.7, color_overlay, 0.3, 0)
save_path = os.path.join("output", f"sam2_box_whight_colored_{img_name}")
# cv2.imwrite(save_path, result)
cv2.imwrite(save_path, color_overlay)
print(f"✅ 结果已保存至: {save_path}")

save_path = os.path.join("cache", "character_mask", "character_mask.png")
cv2.imwrite(save_path, character_mask)
print(f"character_mask已保存至: {save_path}")





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
for i, item in enumerate(ocr_results):
    text = item["text"]
    center = item["center"] # [x, y]
    
    # 解决“只涂文字”的关键点：使用多掩码输出并选择最大范围
    masks, scores, _ = sam2_engine.predictor.predict(
        point_coords=np.array([center]),
        point_labels=np.array([1]),
        multimask_output=True # 必须为 True
    )
    
    best_mask = masks[np.argmax(scores)]
    
    # --- 新增：保存单个掩码到 cache ---
    # 1. 将布尔矩阵转换为黑白图像 (白色 255 代表区域)
    mask_image = (best_mask.astype(np.uint8)) * 255
    
    # 2. 清理文件名中的特殊字符，防止报错
    safe_text = "".join(x for x in text if x.isalnum() or x in "._- ")
    mask_filename = f"{i}_{safe_text}.png"

    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    cache_dir = os.path.join(root_dir, "cache")

    mask_save_path = os.path.join(cache_dir, mask_filename)
    
    # 3. 保存图片
    cv2.imwrite(mask_save_path, mask_image)
    print(f"mask saved to path: {mask_save_path}")


    # 选择得分最高或范围最广的 mask (通常 masks[2] 范围最大)
    # 你可以根据实际效果尝试 masks[np.argmax(scores)] 或固定 masks[1]
    best_mask = masks[np.argmax(scores)].astype(bool)
    
    # 随机颜色并上色
    color = np.random.randint(0, 255, (3,)).tolist()
    color_overlay[best_mask] = color
    print(f"已从 JSON 读取并上色: {text}")

# --- 3. 合成保存 ---
final_res = cv2.addWeighted(image_bgr, 0.7, color_overlay, 0.3, 0)
save_path = os.path.join("output", f"point_colored_{img_name}")
cv2.imwrite(save_path, final_res)
print(f"picture saved to: {save_path}")


#-------------------------------------------------------------
#SAM结束，可视化开始
#-------------------------------------------------------------






CACHE_DIR = './cache'        # 存放 Mask 的文件夹
OUTPUT_DIR = './output'      # 结果输出文件夹
# SOURCE_IMG = img_name   # 原始底图路径 (请确保文件存在)

# --- 2. 准备数据 (关键步骤) ---
# 这里假设你已经有了 boxes 数据
# ⚠️重要：boxes 的顺序必须与文件名数字排序后的 mask 顺序完全一致！
# 例如：boxes[0] 对应 1.png, boxes[1] 对应 2.png

# 示例数据：假设文件夹里有3个mask，这里就需要3个box
boxes = boxes_copy


# ================= Workflow Start =================

try:
    # Step 1: 实例化过滤器并计算重叠
    print(">>> [1/2] Analyzing Mask Overlaps...")
    overlap_filter = MaskOverlapFilter(cache_dir=CACHE_DIR, threshold=0.6)
    
    # 获取布尔数组 [True, False, True, ...]
    result_flags = overlap_filter.check_overlaps()
    
    print(f"Flags Result: {result_flags}")
    print(f"Count: {len(result_flags)} (True={sum(result_flags)}, False={len(result_flags)-sum(result_flags)})")


    # Step 2: 校验数据对齐
    # 这是一个常见的坑，如果 mask 文件数量和 boxes 数量对不上，可视化会报错
    if len(result_flags) != len(boxes):
        raise ValueError(f"Data Mismatch! Found {len(result_flags)} masks but provided {len(boxes)} boxes.")


    # Step 3: 实例化可视化器并生成图像
    print("\n>>> [2/2] Visualizing Results...")
    visualizer = ResultVisualizer(cache_dir=CACHE_DIR, output_dir=OUTPUT_DIR)
    
    saved_path = visualizer.draw_and_save(
        image_path=img_path,
        boxes=boxes,               # 传入原始框
        result_flags=result_flags, # 传入上一步计算的 Flag
        output_filename='processed_' + img_name 
    )

    print(f"\n✅ All Done! Output saved to: {os.path.abspath(saved_path)}")

    print("\n>>> [3/3] Cleaning up cache (.png)...")
    if os.path.exists(CACHE_DIR):
        deleted_count = 0
        for filename in os.listdir(CACHE_DIR):
            # 只删除 .png 文件，防止误删其他文件
            if filename.lower().endswith('.png'):
                file_path = os.path.join(CACHE_DIR, filename)
                try:
                    os.remove(file_path)
                    deleted_count += 1
                except OSError as e:
                    print(f"⚠️ Failed to delete {filename}: {e}")
        
        print(f"🗑️  Cleanup complete. Removed {deleted_count} files.")
    else:
        print("Cache directory does not exist, nothing to clean.")

except FileNotFoundError as e:
    print(f"❌ File Error: {e}")
except ValueError as e:
    print(f"❌ Data Error: {e}")
except Exception as e:
    print(f"❌ Unexpected Error: {e}")
    import traceback
    traceback.print_exc()



#-------------------------------------------------------------
#写回文字
#-------------------------------------------------------------


restorer = TextRestorer()
try:
    # 请确保 ./data/factory.jpg 和 ./output/processed_factory.jpg 存在
    restorer.run(img_name)
except Exception as e:
    print(f"Error: {e}")



#-------------------------------------------------------------
#目前没用，以后不知道有没有用
#-------------------------------------------------------------


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