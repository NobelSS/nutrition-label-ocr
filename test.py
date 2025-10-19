import json
from openocr import OpenOCR

onnx_engine = OpenOCR(backend='onnx', device='cpu', onnx_rec_model_path='./model/rec/repsvtr_ch.onnx')
img_path = 'dataset/labeled/image_100.jpg'
result, elapse = onnx_engine(img_path)

raw = result[0]

_, json_str = raw.split('\t', 1)

data = json.loads(json_str)

texts = [item['transcription'] for item in data]
texts = "\n".join(texts)

print("OCR Result: ", texts)
