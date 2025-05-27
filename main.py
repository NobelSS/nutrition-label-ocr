import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from object_detection import detect_object
from deskew import deskew
from perspective_correction import perspective_correction
from preprocess import preprocess
from ocr import use_tesseract

image_path = "dataset/real/20250510_182813.jpg"

cropped = detect_object(image_path, show_result=False, draw_boxes=False)
rectified = perspective_correction(cropped, show_result=False, debug=False)
deskewed = deskew(rectified, show_result=False)
output = preprocess(deskewed, debug=True)
ocr_text = use_tesseract(output)

print(f'\nDetected Text: \n {ocr_text}')