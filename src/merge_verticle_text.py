import numpy as np

def merge_ocr_to_centers(texts, boxes, x_overlap_ratio=0.8):
    """
    合并纵向排列且横向对齐的文字块，并返回文字及其合并后的中心坐标列表
    :param texts: 识别出的文字列表
    :param boxes: 对应位置列表 [x1, y1, x2, y2]
    :param x_overlap_ratio: 横向重合度阈值
    :return: merged_texts (列表), center_coords (嵌套列表，如 [[x,y], [x,y]])
    """
    if not texts:
        return [], []

    # 数据预处理并按 y_min 排序
    combined = []
    for i in range(len(texts)):
        combined.append({
            'text': texts[i],
            'box': list(boxes[i])
        })
    
    combined.sort(key=lambda x: x['box'][1])

    merged_list = []
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
            is_y_overlapped = not (curr_box[3] < next_box[1] or curr_box[1] > next_box[3])

            if is_x_aligned and is_y_overlapped:
                curr_text += next_text
                # 更新坐标并集
                curr_box[0] = min(curr_box[0], next_box[0])
                curr_box[1] = min(curr_box[1], next_box[1])
                curr_box[2] = max(curr_box[2], next_box[2])
                curr_box[3] = max(curr_box[3], next_box[3])
                used_indices.add(j)
        
        # --- 重点修改部分：使用列表 [x, y] 作为坐标表示 ---
        center_x = (curr_box[0] + curr_box[2]) // 2
        center_y = (curr_box[1] + curr_box[3]) // 2
        
        merged_list.append({
            'text': curr_text, 
            'center': [center_x, center_y]  # 明确使用列表
        })
        used_indices.add(i)

    merged_texts = [m['text'] for m in merged_list]
    center_coords = [m['center'] for m in merged_list]
    
    return merged_texts, center_coords

# --- 测试示例 ---
if __name__ == "__main__":
    res_texts = ["门房", "原料"]
    res_boxes = [[106, 29, 160, 61], [808, 32, 860, 63]]

    # 假设由于分行导致的示例
    test_texts = ["生产", "车间", "办公", "室"]
    test_boxes = [
        [100, 100, 200, 150], # 生产
        [105, 140, 195, 190], # 车间 (纵向与生产有重叠，横向对齐)
        [300, 100, 400, 150], # 办公
        [310, 160, 390, 210]  # 室 (纵向不重叠)
    ]

    m_texts, m_boxes = merge_ocr_to_centers(test_texts, test_boxes)
    print("合并后文字:", m_texts)
    print("合并后boxes：", m_boxes)