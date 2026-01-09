from paddleocr import PaddleOCR
from merge_verticle_text import merge_ocr_blocks

img_path = "./data/lQDPKdU6o7Q9-UPNAfTNBIKwsI-AMOPGxKwICjTC7zs2AA_1154_500.jpg"

# 初始化 PaddleOCR 实例
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False)

# 对示例图像执行 OCR 推理 
result = ocr.predict(
    input=img_path)

# print(result[0])

# 可视化结果并保存 json 结果
for res in result:
    res.print()
    res.save_to_img("output")
    res.save_to_json("output")

print(result[0]["rec_texts"])
print(result[0]["rec_boxes"])

texts_merged, boxes_merged = merge_ocr_blocks(texts= result[0]["rec_texts"], boxes= result[0]["rec_boxes"])

print(texts_merged)
print(boxes_merged)

# result_merged = merge_vertical_text(result[0])

# for res in result_merged:
#     res.print()
#     res.save_to_img("output")
#     res.save_to_json("output")



