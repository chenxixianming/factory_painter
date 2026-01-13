import os
import cv2
import numpy as np
from paddleocr import PaddleOCR
# 导入你之前的合并逻辑函数，或者直接把函数写在这个类里
from merge_verticle_1 import merge_ocr_to_centers

import json
import requests

class IndustrialOCRManager:
    def __init__(self, lang='ch'):
        """初始化 OCR 引擎"""
        print("--- 正在初始化 PaddleOCR 引擎 ---")
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang=lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False
        )

        # --- ⬇️ 请补全这部分配置 ⬇️ ---
        self.api_url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
        # 使用你刚才提供的 API Key
        self.api_key = "cc03e248-3c76-4216-838e-2944190cdb3a" 
        # 使用你刚才提供的 Model ID
        self.model_id = "doubao-seed-1-6-250615" 
        # -------------------------------

    def _merge_boxes_by_ids(self, original_boxes, source_ids):
        """
        辅助函数：根据 ID 列表合并多个 Box，计算出新的大 Box 和中心点
        """
        if not source_ids:
            return None, None
            
        selected_boxes = [original_boxes[i] for i in source_ids if i < len(original_boxes)]
        
        if not selected_boxes:
            return None, None

        # 将所有 box 转为 numpy 数组方便计算
        boxes_np = np.array(selected_boxes) # Shape: (N, 4)
        
        # 计算外接矩形 (Union Box)
        # x1, y1 取最小值，x2, y2 取最大值
        new_x1 = np.min(boxes_np[:, 0])
        new_y1 = np.min(boxes_np[:, 1])
        new_x2 = np.max(boxes_np[:, 2])
        new_y2 = np.max(boxes_np[:, 3])
        
        new_box = [float(new_x1), float(new_y1), float(new_x2), float(new_y2)]
        
        # 计算新中心点
        new_center = [
            (new_x1 + new_x2) / 2.0,
            (new_y1 + new_y2) / 2.0
        ]
        
        return new_box, new_center

    def get_ocr_results(self, img_path, verticle_merge = False):
        """
        执行识别并返回合并后的文字与中心点
        """
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"无法找到图像文件: {img_path}")

        # 1. 原始推理
        print(f"正在识别图像: {os.path.basename(img_path)}")
        result = self.ocr.predict(input=img_path)
        
        if not result or len(result[0]["rec_texts"]) == 0:
            return [], []

        raw_texts = result[0]["rec_texts"]
        raw_boxes = result[0]["rec_boxes"]

        # 2. 调用你之前优化的水平扫描合并逻辑
        # 内部会自动读取 img_path 进行墙线探测
        # texts_merged, centers_merged = merge_with_line_scan(
        #     texts=raw_texts, 
        #     boxes=raw_boxes, 
        #     image_path=img_path
        # )
        if verticle_merge == True:
            texts_merged, centers_merged, boxes_merged = merge_ocr_to_centers(
                texts= raw_texts, 
                boxes= raw_boxes)
            print(f"识别完成，原始块: {len(raw_texts)} --> 合并后：{len(texts_merged)}")
            return texts_merged, centers_merged, boxes_merged
        
        print(f"识别完成，原始块: {len(raw_texts)}")
        return raw_texts, raw_boxes
    

    def correct_structure_with_llm(self, texts, boxes):
        """
        调用 LLM 进行语义合并，同时考虑 Box 的空间邻近性
        """
        # if not texts or not boxes:
        #     return [], [], []

        if len(texts) == 0 or len(boxes) == 0:
            return [], [], []

        print("🤖 正在调用豆包 LLM 进行结构化修正...")

        # 1. 构造带有 ID 和 Box 的输入数据
        input_data = []
        for i, text in enumerate(texts):
            # 将 numpy 数组转为 list，并保留整数以减少 token 消耗（如果不需要极高精度）
            box = [int(b) for b in boxes[i]]
            input_data.append({
                "id": i,
                "text": text,
                "box": box  # 增加坐标信息 [x1, y1, x2, y2]
            })

        # 2. 构造升级版 Prompt
        # 核心修改：增加了关于坐标 (Box) 的约束说明
        system_prompt = (
            "你是一个工业图纸 OCR 后处理专家。我会给你一个列表，包含 ID、文字内容 (text) 和 边界框坐标 (box: [xmin, ymin, xmax, ymax])。\n"
            "你的任务是合并被错误切分的词条，但必须同时满足 **语义通顺** 和 **空间临近** 两个条件。\n\n"
            "请遵循以下规则：\n"
            "1. **空间约束（最重要）**：只有当两个框在垂直空间上非常接近且在水平空间上有大幅度重合时，才允许合并。如果两个框的坐标相差很远（例如垂直坐标相差超过一定值或水平坐标完全错开），绝对不要合并，即使它们语义上有关联。\n"
            "2. **语义合并**：在满足空间约束的前提下，合并被切断的词（例如 ['生', '产车间'] -> ['生产车间']）。注意有些字可能有OCR识别错误，需要修改错误才会出现明显的语义联系。另外，只有单个文字的词条，如果这个文字的字形比较复杂，那么几乎可以肯定这个词条和另外某个词条需要合并，如果没有发现语义联系可以先尝试合并再判断是否可能是OCR识别错误\n"
            "3. **纠错**：修正明显的 OCR 错误。\n"
            "4. **删除无意义数据**：删除完全无意义的词条，例如纯数字。如果不能完全肯定某词条无意义，予以保留。\n"
            "5. **返回格式**：严格只返回一个 JSON 列表，每个对象包含：\n"
            "   - 'text': 修正/合并后的文本\n"
            "   - 'source_ids': 该文本对应的原始 ID 列表（按阅读顺序排列）。\n\n"
            "示例：\n"
            "输入: [{'id':0, 'text':'生', 'box':[10,10,20,20]}, {'id':1, 'text':'产', 'box':[22,10,32,20]}, {'id':2, 'text':'室', 'box':[100,100,120,120]}]\n"
            "输出: [{'text':'生产', 'source_ids':[0, 1]}, {'text':'室', 'source_ids':[2]}]\n"
            "(解释: id 0 和 1 坐标紧邻且语义连贯，故合并；id 2 距离太远，虽有'生产室'这个词，但不应合并。)"
        )

        user_content = f"待处理数据: {json.dumps(input_data, ensure_ascii=False)}"

        # 3. 调用 API
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.1 # 保持低温，避免胡乱联想
        }
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }

        try:
            response = requests.post(
                self.api_url, 
                headers=headers, 
                json=payload, 
                timeout=600)
            
            response.raise_for_status()
            
            content = response.json()['choices'][0]['message']['content']
            content = content.replace("```json", "").replace("```", "").strip()
            
            llm_result = json.loads(content)
            
            # 4. 解析并重组数据 (这部分逻辑不变)
            new_texts = []
            new_boxes = []
            new_centers = []

            for item in llm_result:
                corrected_text = item.get("text")
                source_ids = item.get("source_ids", [])
                
                # 调用之前的辅助函数计算合并后的 Box
                merged_box, merged_center = self._merge_boxes_by_ids(boxes, source_ids)
                
                if merged_box and merged_center:
                    new_texts.append(corrected_text)
                    new_boxes.append(merged_box)
                    new_centers.append(merged_center)
            
            print(f"✅ 结构化修正完成 (含空间约束): {len(texts)} -> {len(new_texts)}")
            return new_texts, new_centers, new_boxes

        except Exception as e:
            print(f"❌ LLM 处理失败: {e}")
            # 回退策略：计算原始中心点返回
            original_centers = []
            for b in boxes:
                original_centers.append([(b[0]+b[2])/2.0, (b[1]+b[3])/2.0])
            return texts, original_centers, boxes