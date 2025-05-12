from object_detection import detect_object
from deskew import deskew

image_path = "./dataset/real/20250510_181112 - Copy.jpg"

cropped_image = detect_object(image_path)
deskewed_image = deskew(cropped_image)