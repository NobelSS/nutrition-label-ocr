import numpy as np
from PIL import Image
import pytesseract
import os
import logging
import json
from openocr import OpenOCR

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

    if image.ndim == 2:
        # Grayscale → stack to 3 channels
        image_bgr = np.stack([image] * 3, axis=-1)
    elif image.ndim == 3:
        if image.shape[2] == 4:
            # RGBA → drop alpha
            image = image[:, :, :3]
        # RGB → convert to BGR
        image_bgr = image[:, :, ::-1]
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}")

    # Perform OCR
    result = ocr.predict(input=image_bgr)

    texts = result[0]['rec_texts']
    
    return "\n".join(texts)

def use_svtrv2_mobile(image: np.ndarray):
    
    if image.ndim == 2:
        # Grayscale → stack to 3 channels
        image_bgr = np.stack([image] * 3, axis=-1)
    elif image.ndim == 3:
        if image.shape[2] == 4:
            # RGBA → drop alpha
            image = image[:, :, :3]
        # RGB → convert to BGR
        image_bgr = image[:, :, ::-1]
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}")
    
    onnx_engine = OpenOCR(
        backend='onnx',
        device='cpu',
        onnx_det_model_path='./model/det/dbnet.onnx',
        onnx_rec_model_path='./model/rec/repsvtr_ch.onnx',
    )
    
    print("Detector model path:", onnx_engine.text_detector.onnx_det_engine.onnx_session._model_path)
    print("Recognizer model path:", onnx_engine.text_recognizer.onnx_rec_engine.onnx_session._model_path)
    
    result, _ = onnx_engine(img_numpy=image_bgr)
    if not result or not isinstance(result[0], list):
        return ""

    data = result[0]

    texts = [item['transcription'] for item in data if 'transcription' in item]

    return " ".join(texts)


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
    elif engine.lower() == 'svtrv2_mobile':
        return use_svtrv2_mobile(image)
    else:
        raise ValueError(f"Unsupported OCR engine: {engine}. Supported engines: 'tesseract', 'paddleocr'")



