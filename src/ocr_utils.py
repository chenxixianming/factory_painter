import os
import cv2
import numpy as np
from paddleocr import PaddleOCR
# 导入你之前的合并逻辑函数，或者直接把函数写在这个类里
from merge_verticle_1 import merge_ocr_to_centers

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

    def get_merged_results(self, img_path):
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
        texts_merged, centers_merged, boxes_merged = merge_ocr_to_centers(
            texts= raw_texts, 
            boxes= raw_boxes)
        
        print(f"识别完成，原始块: {len(raw_texts)} -> 合并后区域: {len(texts_merged)}")
        return texts_merged, centers_merged, boxes_merged