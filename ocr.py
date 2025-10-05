import numpy as np
from PIL import Image
import pytesseract
import os
import logging

# Suppress PaddleOCR verbose logging
os.environ['FLAGS_allocator_strategy'] = 'auto_growth'
os.environ['FLAGS_fraction_of_gpu_memory_to_use'] = '0.5'

# Set logging level to reduce verbose output
logging.getLogger('ppocr').setLevel(logging.WARNING)
logging.getLogger('paddle').setLevel(logging.WARNING)

from paddleocr import PaddleOCR


def use_tesseract(image: np.ndarray):
    pil_image = Image.fromarray(image)

    custom_config = r'--oem 3 --psm 6'  # OEM 3: Default, PSM 6: Assume a uniform block of text

    text = pytesseract.image_to_string(pil_image, config=custom_config)

    return text

def use_paddleocr(image: np.ndarray, lang='en'):

    # Initialize PaddleOCR with reduced verbosity
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=True,
        use_textline_orientation=False)

    if len(image.shape) == 3 and image.shape[2] == 3:
        # Assume input is RGB, convert to BGR for PaddleOCR
        image_bgr = image[:, :, ::-1]
    else:
        # Grayscale image, convert to BGR
        image_bgr = np.stack([image, image, image], axis=-1)

    # Perform OCR
    
    result = ocr.predict(input=image_bgr)
    
    for res in result:
        # res.print()
        res.save_to_img("output")
        res.save_to_json("output")
   

    return res

def perform_ocr(image: np.ndarray, engine='tesseract', **kwargs):
    """
    Perform OCR using the specified engine

    Args:
        image: Input image as numpy array
        engine: OCR engine to use ('tesseract' or 'paddleocr')
        **kwargs: Additional arguments for the OCR engine

    Returns:
        Extracted text as string
    """
    if engine.lower() == 'tesseract':
        return use_tesseract(image)
    elif engine.lower() == 'paddleocr':
        lang = kwargs.get('lang', 'en')
        return use_paddleocr(image, lang=lang)
    else:
        raise ValueError(f"Unsupported OCR engine: {engine}. Supported engines: 'tesseract', 'paddleocr'")



