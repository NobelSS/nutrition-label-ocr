import numpy as np
from PIL import Image
import pytesseract

def use_tesseract(image: np.ndarray):
    print("Performing OCR using Tesseract...")
    pil_image = Image.fromarray(image)
    
    custom_config = r'--oem 3 --psm 6'  # OEM 3: Default, PSM 6: Assume a uniform block of text

    text = pytesseract.image_to_string(pil_image, config=custom_config)
    
    return text


