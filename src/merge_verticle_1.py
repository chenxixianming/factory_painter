import numpy as np

def merge_ocr_to_centers(texts, boxes, x_overlap_ratio=0.8):
    """
    合并纵向排列且横向对齐的文字块
    :param texts: 识别出的文字列表
    :param boxes: 对应位置列表 [x1, y1, x2, y2]
    :param x_overlap_ratio: 横向重合度阈值
    :return: 
        merged_texts: 合并后的文字列表
        center_coords: 合并后的中心坐标列表 [[x, y], ...]
        merged_boxes: 合并后的外接矩形列表 [[x1, y1, x2, y2], ...]
    """
    if not texts:
        return [], [], []

    # 数据预处理并按 y_min 排序
    combined = []
    for i in range(len(texts)):
        combined.append({
            'text': texts[i],
            'box': list(boxes[i]) # 保持 [x1, y1, x2, y2]
        })
    
    combined.sort(key=lambda x: x['box'][1])

    merged_results = []
    used_indices = set()

    for i in range(len(combined)):
        if i in used_indices:
            continue
        
        curr_text = combined[i]['text']
        curr_box = combined[i]['box']
        
        for j in range(i + 1, len(combined)):
            if j in used_indices:
                continue
            
            next_text = combined[j]['text']
            next_box = combined[j]['box']

            # 1. 计算 X 轴重合宽度
            x_overlap_min = max(curr_box[0], next_box[0])
            x_overlap_max = min(curr_box[2], next_box[2])
            overlap_width = max(0, x_overlap_max - x_overlap_min)

            width_i = curr_box[2] - curr_box[0]
            width_j = next_box[2] - next_box[0]
            min_width = min(width_i, width_j)

            # 2. 判断横向对齐与纵向重叠
            is_x_aligned = overlap_width >= (min_width * x_overlap_ratio)
            # 判断 y 轴是否有接触或重叠（用于合并分行文字）
            is_y_overlapped = not (curr_box[3] < next_box[1] or curr_box[1] > next_box[3])

            if is_x_aligned and is_y_overlapped:
                curr_text += next_text
                # 更新当前合并块的坐标范围（并集）
                curr_box[0] = min(curr_box[0], next_box[0])
                curr_box[1] = min(curr_box[1], next_box[1])
                curr_box[2] = max(curr_box[2], next_box[2])
                curr_box[3] = max(curr_box[3], next_box[3])
                used_indices.add(j)
        
        # 计算合并后的中心点
        # 建议使用浮点数，方便后续 JSON 存储和精准擦除
        center_x = (curr_box[0] + curr_box[2]) / 2.0
        center_y = (curr_box[1] + curr_box[3]) / 2.0
        
        merged_results.append({
            'text': curr_text, 
            'center': [center_x, center_y],
            'box': [float(curr_box[0]), float(curr_box[1]), float(curr_box[2]), float(curr_box[3])]
        })
        used_indices.add(i)

    # 拆分结果
    merged_texts = [m['text'] for m in merged_results]
    center_coords = [m['center'] for m in merged_results]
    merged_boxes = [m['box'] for m in merged_results]
    
    return merged_texts, center_coords, merged_boxes

# --- 测试示例 ---
if __name__ == "__main__":
    test_texts = ["生产", "车间", "办公", "室"]
    test_boxes = [
        [100, 100, 200, 150], # 生产
        [105, 140, 195, 190], # 车间 (纵向重叠)
        [300, 100, 400, 150], # 办公
        [310, 145, 390, 210]  # 室 (纵向重叠)
    ]

    m_texts, m_centers, m_boxes = merge_ocr_to_centers(test_texts, test_boxes)
    
    print("--- 合并结果 ---")
    for t, c, b in zip(m_texts, m_centers, m_boxes):
        print(f"文字: {t}")
        print(f"  中心: {c}")
        print(f"  Box:  {b}")