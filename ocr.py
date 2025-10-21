import numpy as np
from PIL import Image
import pytesseract
import os
import logging
import json
from openocr import OpenOCR, OpenRecognizer
from paddleocr import TextDetection, PaddleOCR

# Suppress PaddleOCR verbose logging
os.environ['FLAGS_allocator_strategy'] = 'auto_growth'
os.environ['FLAGS_fraction_of_gpu_memory_to_use'] = '0.5'

# Set logging level to reduce verbose output
logging.getLogger('ppocr').setLevel(logging.WARNING)
logging.getLogger('paddle').setLevel(logging.WARNING)

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

def use_svtrv2_mobile_with_paddleocr_detection(image: np.ndarray):
    
    text_detector = TextDetection()
    det_result = text_detector.predict(image)
        
    boxes = det_result[0]['dt_polys']
    
    # Initialize SVTR recognizer
    recognizer = OpenRecognizer(
        backend='onnx',
        onnx_model_path='./model/rec/repsvtr_ch.onnx',
        
    )
    
    texts = []
    img_height, img_width = image.shape[:2]
    
    for idx, box in enumerate(boxes):
        try:
            # Crop region with bounds checking
            x_coords = [point[0] for point in box]
            y_coords = [point[1] for point in box]
            
            x_min = max(0, int(min(x_coords)))
            x_max = min(img_width, int(max(x_coords)))
            y_min = max(0, int(min(y_coords)))
            y_max = min(img_height, int(max(y_coords)))
            
            cropped = image[y_min:y_max, x_min:x_max]
                        
            # Skip if cropped region is empty
            if cropped.size == 0:
                print(f"Box {idx}: Empty crop - skipping")
                continue
            
            # Recognize text
            rec_result = recognizer(img_numpy=cropped)
            # print(f"Box {idx}: Recognition result = {rec_result}")
            
            # Handle different result formats
            if rec_result:
                
                texts.append(rec_result[0]['text'])
                
                # if text and text.strip():
                #     texts.append(text.strip())
                #     print(f"Box {idx}: Added text '{text.strip()}'")
        
        except Exception as e:
            print(f"Box {idx}: Error processing - {str(e)}")
            continue
    
    return "\n".join(texts)

def use_svtrv2_mobile_with_easyocr_detection(image: np.ndarray):
    import easyocr
    
    # Initialize EasyOCR for detection only
    reader = easyocr.Reader(['en'], recognizer=False, gpu=False)
    
    if image.ndim == 2:
        image_rgb = np.stack([image] * 3, axis=-1)
    elif image.ndim == 3:
        if image.shape[2] == 4:
            image_rgb = image[:, :, :3]
        else:
            image_rgb = image
    
    # Detect text regions
    horizontal_list, free_list = reader.detect(image_rgb)
    
    # Initialize SVTR recognizer
    recognizer = OpenRecognizer(
        backend='onnx',
        onnx_model_path='./model/rec/repsvtr_ch.onnx'
    )
    
    texts = []
    img_height, img_width = image_rgb.shape[:2]
    
    for idx, box in enumerate(horizontal_list[0]):
        try:
            x_min, x_max, y_min, y_max = map(int, box)
            
            # Bounds checking
            x_min = max(0, x_min)
            x_max = min(img_width, x_max)
            y_min = max(0, y_min)
            y_max = min(img_height, y_max)
            
            # Skip invalid boxes
            # if x_max <= x_min or y_max <= y_min:
            #     continue
                
            # if (x_max - x_min) < 5 or (y_max - y_min) < 5:
            #     continue
            
            cropped = image_rgb[y_min:y_max, x_min:x_max]
            
            if cropped.size == 0:
                continue
            
            # Convert to BGR for consistency
            cropped_bgr = cropped[:, :, ::-1]
            
            # Recognize text
            rec_result = recognizer(img_numpy=cropped_bgr)
            
            # Handle different result formats
            if rec_result:
               texts.append(rec_result[0]['text'])
                    
        except Exception as e:
            print(f"Box {idx}: Error - {str(e)}")
            continue
    
    return "\n".join(texts)

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
    
    if image.ndim == 2:
        # Grayscale → stack to 3 channels
        image = np.stack([image] * 3, axis=-1)
    elif image.ndim == 3:
        if image.shape[2] == 4:
            # RGBA → drop alpha
            image = image[:, :, :3]
        # RGB → convert to BGR
        image = image[:, :, ::-1]
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}")
    
    if engine.lower() == 'tesseract':
        return use_tesseract(image)
    elif engine.lower() == 'paddleocr':
        lang = kwargs.get('lang', 'en')
        return use_paddleocr(image, lang=lang)
    elif engine.lower() == 'svtrv2_mobile':
        # return use_svtrv2_mobile_with_paddleocr_detection(image)
        return use_svtrv2_mobile_with_easyocr_detection(image)
    else:
        raise ValueError(f"Unsupported OCR engine: {engine}. Supported engines: 'tesseract', 'paddleocr', 'svtrv2_mobile'")



