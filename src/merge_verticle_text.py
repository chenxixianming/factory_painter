import numpy as np

def merge_ocr_blocks(texts, boxes, x_overlap_ratio=0.8):
    """
    合并纵向排列且横向高度重合的文字块
    :param texts: 识别出的文字列表
    :param boxes: 对应位置列表 [x1, y1, x2, y2]
    :param x_overlap_ratio: 横向重合度阈值 (默认80%)
    """
    if not texts:
        return [], []

    # 将数据转化为易处理的格式，并按 y_min 排序
    combined = []
    for i in range(len(texts)):
        combined.append({
            'text': texts[i],
            'box': list(boxes[i])
        })
    
    # 按纵向起始位置排序，确保是从上往下检查
    combined.sort(key=lambda x: x['box'][1])

    merged_list = []
    used_indices = set()

    for i in range(len(combined)):
        if i in used_indices:
            continue
        
        curr_text = combined[i]['text']
        curr_box = combined[i]['box']
        
        # 尝试寻找后续可以合并的块
        for j in range(i + 1, len(combined)):
            if j in used_indices:
                continue
            
            next_text = combined[j]['text']
            next_box = combined[j]['box']

            # 1. 计算横向(X轴)重合部分长度
            x_overlap_min = max(curr_box[0], next_box[0])
            x_overlap_max = min(curr_box[2], next_box[2])
            overlap_width = max(0, x_overlap_max - x_overlap_min)

            # 计算两个块各自的宽度
            width_i = curr_box[2] - curr_box[0]
            width_j = next_box[2] - next_box[0]
            min_width = min(width_i, width_j)

            # 2. 检查横向重合度是否超过 80%
            is_x_aligned = overlap_width >= (min_width * x_overlap_ratio)

            # 3. 检查纵向(Y轴)是否有重叠
            # 只要 A 的顶部小于 B 的底部 且 A 的底部大于 B 的顶部，即为有重叠
            is_y_overlapped = not (curr_box[3] < next_box[1] or curr_box[1] > next_box[3])

            if is_x_aligned and is_y_overlapped:
                # 执行合并
                curr_text += next_text # 这里可以根据需要加换行符 "\n"
                # 更新坐标框为两者的并集 (外接矩形)
                curr_box[0] = min(curr_box[0], next_box[0]) # x_min
                curr_box[1] = min(curr_box[1], next_box[1]) # y_min
                curr_box[2] = max(curr_box[2], next_box[2]) # x_max
                curr_box[3] = max(curr_box[3], next_box[3]) # y_max
                
                used_indices.add(j)
        
        merged_list.append({'text': curr_text, 'box': curr_box})
        used_indices.add(i)

    # 拆分回原格式输出
    merged_texts = [m['text'] for m in merged_list]
    merged_boxes = [m['box'] for m in merged_list]
    
    return merged_texts, merged_boxes

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

    m_texts, m_boxes = merge_ocr_blocks(test_texts, test_boxes)
    print("合并后文字:", m_texts)
    print("合并后boxes：", m_boxes)