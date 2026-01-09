import numpy as np
import cv2

def has_horizontal_wall_between(image_rgb, box_above, box_below, wall_threshold=50, occupancy_ratio=0.8):
    """
    从上方文字中心向下扫描至下方文字中心，检测是否存在水平长线（墙）
    """
    # 1. 计算两个 box 的几何中心 Y 坐标
    center_y_above = int((box_above[1] + box_above[3]) // 2)
    center_y_below = int((box_below[1] + box_below[3]) // 2)
    
    # 2. 确定扫描的横向范围（以宽度较窄的那个为准或固定使用上方宽度，这里沿用上方宽度）
    x_start = int(box_above[0])
    x_end = int(box_above[2])
    scan_width = x_end - x_start
    
    if scan_width <= 0 or center_y_below <= center_y_above:
        return False

    # 3. 截取从“上中心”到“下中心”的垂直条带
    # 注意：OpenCV 坐标是 [y_start:y_end, x_start:x_end]
    roi = image_rgb[center_y_above:center_y_below, x_start:x_end]
    
    # 4. 转换为亮度（灰度）并计算每一行的黑像素占比
    # 使用均值法快速获取亮度
    brightness = np.mean(roi, axis=2) 
    
    # 统计每一行中亮度低于阈值的像素个数
    black_pixel_counts = np.sum(brightness < wall_threshold, axis=1)
    
    # 计算每一行的最大占比
    # 如果扫描区域内任意一行连续黑色像素占比超过阈值，则判定有墙
    max_occupancy = np.max(black_pixel_counts) / scan_width
    
    return max_occupancy >= occupancy_ratio

def merge_with_line_scan(texts, boxes, image_path, x_overlap_ratio=0.8):
    """
    采用水平线段扫描逻辑的合并函数
    """
    if not texts: return [], []
    
    img = cv2.imread(image_path)
    if img is None: return texts, [[(b[0]+b[2])//2, (b[1]+b[3])//2] for b in boxes]

    combined = []
    for i in range(len(texts)):
        combined.append({'text': texts[i], 'box': list(boxes[i])})
    
    # 按 Y 轴排序
    combined.sort(key=lambda x: x['box'][1])
    
    merged_list = []
    used_indices = set()

    for i in range(len(combined)):
        if i in used_indices: continue
        curr_text = combined[i]['text']
        curr_box = combined[i]['box']
        
        for j in range(i + 1, len(combined)):
            if j in used_indices: continue
            
            next_box = combined[j]['box']
            
            # 基础几何条件判断
            w1 = curr_box[2] - curr_box[0]
            h1 = curr_box[3] - curr_box[1]
            w2 = next_box[2] - next_box[0]
            h2 = next_box[3] - next_box[1]
            std_val = min(w1, h1, w2, h2)
            
            dist_y = next_box[1] - curr_box[3]
            x_overlap = max(0, min(curr_box[2], next_box[2]) - max(curr_box[0], next_box[0]))
            is_x_aligned = x_overlap >= (min(w1, w2) * x_overlap_ratio)

            # 只有满足几何邻近时，才启动耗时的“线段扫描”
            if is_x_aligned and dist_y < std_val:
                # 执行新的扫描逻辑
                if not has_horizontal_wall_between(img, curr_box, next_box):
                    # 认为没有墙，执行合并...
                    curr_text += combined[j]['text']
                    curr_box = [min(curr_box[0], next_box[0]), min(curr_box[1], next_box[1]),
                                max(curr_box[2], next_box[2]), max(curr_box[3], next_box[3])]
                    used_indices.add(j)

        merged_list.append({'text': curr_text, 'center': [(curr_box[0]+curr_box[2])//2, (curr_box[1]+curr_box[3])//2]})
        used_indices.add(i)

    return [m['text'] for m in merged_list], [m['center'] for m in merged_list]

if __name__ == "__main__":
    import os

    # 1. 创建测试用的模拟图像 (500x500 白色背景)
    test_img_path = "debug_test_canvas.png"
    canvas = np.ones((500, 500, 3), dtype=np.uint8) * 255

    # --- 模拟场景 A: 属于同一区域的两行字 (应合并) ---
    # 文字内容: "生产", "车间"
    texts_a = ["生产", "车间"]
    boxes_a = [
        [100, 100, 200, 130], # "生产" 的 box
        [100, 140, 200, 170]  # "车间" 的 box (垂直间距 10px)
    ]

    # --- 模拟场景 B: 被墙隔开的两个区域 (不应合并) ---
    # 文字内容: "男厕", "女厕"
    texts_b = ["男厕", "女厕"]
    boxes_b = [
        [300, 100, 400, 130], # "男厕"
        [300, 150, 400, 180]  # "女厕"
    ]
    # 在 B 场景中间画一条“墙线” (y=140 处)
    # 长度为 100px (300 到 400)，完全覆盖了文字宽度
    cv2.line(canvas, (300, 142), (400, 142), (0, 0, 0), 2) 

    # 保存模拟图
    cv2.imwrite(test_img_path, canvas)

    # 合并测试数据
    all_texts = texts_a + texts_b
    all_boxes = boxes_a + boxes_b

    print(f"--- 开始算法测试 ---")
    print(f"输入原始文字块数量: {len(all_texts)}")

    # 2. 调用你的合并函数
    merged_texts, merged_centers = merge_with_line_scan(all_texts, all_boxes, test_img_path)

    # 3. 验证结果
    print("\n[测试结果]")
    for t, c in zip(merged_texts, merged_centers):
        print(f"识别到区域: {t: <10} | 中心坐标: {c}")

    # 预期检查逻辑
    if "生产车间" in merged_texts and "男厕" in merged_texts and "女厕" in merged_texts:
        print("\n✅ 测试通过：'生产车间' 已成功合并，且'男厕'与'女厕'被墙线正确阻断！")
    else:
        print("\n❌ 测试失败：请检查 occupancy_ratio 或 wall_threshold 设置。")

    # 4. 可视化导出 (辅助观察)
    debug_view = cv2.imread(test_img_path)
    for center in merged_centers:
        cv2.circle(debug_view, (center[0], center[1]), 5, (0, 0, 255), -1)
    cv2.imwrite("debug_test_result.png", debug_view)
    print(f"\n可视化调试图已保存至: debug_test_result.png")
    
    # 清理测试原图
    if os.path.exists(test_img_path):
        os.remove(test_img_path)