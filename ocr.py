import numpy as np
from PIL import Image
import pytesseract
import os
import logging
import json
from openocr import OpenOCR, OpenRecognizer
from paddleocr import TextDetection, PaddleOCR
import cv2
import matplotlib.pyplot as plt

# Suppress PaddleOCR verbose logging
os.environ['FLAGS_allocator_strategy'] = 'auto_growth'
os.environ['FLAGS_fraction_of_gpu_memory_to_use'] = '0.5'

# Set logging level to reduce verbose output
logging.getLogger('ppocr').setLevel(logging.WARNING)
logging.getLogger('paddle').setLevel(logging.WARNING)

def draw_detection_boxes(image: np.ndarray, detections: list, output_path: str = None, 
                        box_color: tuple = (0, 0, 255), text_color: tuple = (0, 255, 0), 
                        font_scale: float = 0.7, thickness: int = 2, line_thickness: int = 1):
    """
    Draw bounding boxes and text labels on an image.
    
    Args:
        image: Input image as numpy array (RGB format)
        detections: List of detection dictionaries with 'points', 'transcription', and 'score' keys
        output_path: Path to save the image with boxes (optional)
        box_color: Color for bounding box lines (BGR format)
        text_color: Color for text labels (BGR format)
        font_scale: Font scale for text labels
        thickness: Thickness of bounding box lines
        line_thickness: Thickness of text lines
    
    Returns:
        Image with drawn bounding boxes (BGR format)
    """
    if not detections:
        return image
    
    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    for det in detections:
        if not isinstance(det, dict):
            continue

        points = det.get("points", [])
        text = det.get("transcription", "")
        score = det.get("score", 0)

        # Validate and normalize points
        if points is None:
            continue
            
        # Handle both numpy arrays and lists
        if hasattr(points, 'tolist'):
            points = points.tolist()
        
        if not points or len(points) == 0:
            continue
            
        if not isinstance(points[0], (list, tuple)):
            continue

        pts = np.array(points, dtype=np.int32)

        # Draw polygon
        cv2.polylines(image_bgr, [pts], isClosed=True, color=box_color, thickness=thickness)

        # Draw text label
        x, y = pts[0]
        cv2.putText(
            image_bgr,
            f"{text} ({score:.2f})",
            (x, max(20, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            line_thickness,
            cv2.LINE_AA,
        )
    
    # Save image if output path is provided
    if output_path:
        cv2.imwrite(output_path, image_bgr)
        print(f"[Detection] Boxes saved to: {output_path}")
    
    return image_bgr

def use_tesseract(image: np.ndarray, draw_boxes: bool = False):
    """
    Perform OCR using Tesseract with optional bounding box visualization.
    
    Args:
        image: Input image as numpy array (RGB format)
        draw_boxes: Whether to draw bounding boxes on the image
    
    Returns:
        Extracted text as string
    """
    pil_image = Image.fromarray(image)

    custom_config = r'--oem 3 --psm 6'  # OEM 3: Default, PSM 6: Assume a uniform block of text

    text = pytesseract.image_to_string(pil_image, config=custom_config)
    
    # Get bounding box data if drawing is requested
    if draw_boxes:
        # Get detailed data including bounding boxes
        data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
        
        detections = []
        for i in range(len(data["text"])):
            if int(data["conf"][i]) > 0 and data["text"][i].strip() != "":
                x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                # Convert rectangle to polygon format (4 corners)
                points = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
                detections.append({
                    'points': points,
                    'transcription': data["text"][i],
                    'score': float(data["conf"][i]) / 100.0  # Convert confidence to 0-1 range
                })
        
        if detections:
            draw_detection_boxes(
                image, detections,
                output_path='tesseract_boxes.png'
            )

    return text

def use_paddleocr(image: np.ndarray, lang='en', draw_boxes: bool = False):
    """
    Perform OCR using PaddleOCR with optional bounding box visualization.
    
    Args:
        image: Input image as numpy array (RGB format)
        lang: Language for OCR ('en', 'ch', etc.)
        draw_boxes: Whether to draw bounding boxes on the image
    
    Returns:
        Extracted text as string
    """
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
    
    # Get bounding box data if drawing is requested
    if draw_boxes:
        detections = []
        if 'dt_polys' in result[0]:
            boxes = result[0]['dt_polys']
            texts_with_boxes = result[0]['rec_texts']
            scores = result[0]['rec_scores'] if 'rec_scores' in result[0] else [1.0] * len(texts_with_boxes)
            
            for i, (box, text, score) in enumerate(zip(boxes, texts_with_boxes, scores)):
                detections.append({
                    'points': box,
                    'transcription': text,
                    'score': float(score)
                })
        
        if detections:
            # Convert BGR back to RGB for drawing function
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            draw_detection_boxes(
                image_rgb, detections,
                output_path='paddleocr_boxes.png'
            )
    
    return "\n".join(texts)

def use_svtrv2_mobile_with_paddleocr_detection(image: np.ndarray):
    
    text_detector = TextDetection()
    det_result = text_detector.predict(image)
        
    boxes = det_result[0]['dt_polys']
    
    # Initialize SVTR recognizer
    recognizer = OpenRecognizer(
        backend='onnx',
        # onnx_model_path='./model/rec/repsvtr_ch.onnx',
        
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
        # onnx_model_path='./model/rec/repsvtr_ch.onnx'
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

def use_svtrv2_mobile(image: np.ndarray, draw_boxes: bool = False):
    """
    Perform OCR using SVTRv2 Mobile model with optional bounding box visualization.
    
    Args:
        image: Input image as numpy array (RGB format)
        draw_boxes: Whether to draw bounding boxes on the image
    
    Returns:
        Extracted text as string
    """
    # ocr = OpenOCR(backend='onnx', device='cpu', onnx_rec_model_path='./model/rec/repsvtr_ch.onnx')
    ocr = OpenOCR(backend='onnx', device='cpu')
    result = ocr(img_numpy=image)
    
    texts = []
    detections = []
        
    # OpenOCR returns a tuple: (detections_list, timing_info)
    if isinstance(result, tuple) and len(result) >= 1:
        detections_list = result[0]
        if isinstance(detections_list, list) and len(detections_list) > 0:
            detections = detections_list[0]  # first item is actual detection list

    if not detections:
        print("[SVTRv2] No detections found.")
        return ""
    # print(f"Detections: {detections}")

    # ✅ Sort boxes in reading order
    detections_sorted = sort_boxes_reading_order(detections)

    # ✅ Extract text in that order
    for det in detections_sorted:
        if isinstance(det, dict) and 'transcription' in det:
            text = det['transcription'].strip()
            if text:
                texts.append(text)
    
    # Draw bounding boxes if requested
    if draw_boxes and detections:
        draw_detection_boxes(
            image, detections, 
            output_path='svtrv2_boxes.png'
        )
    
    return "\n".join(texts)

def perform_ocr(image: np.ndarray, engine='tesseract', draw_boxes: bool = False, **kwargs):
    """
    Perform OCR using the specified engine with optional bounding box visualization.

    Args:
        image: Input image as numpy array
        engine: OCR engine to use ('tesseract', 'paddleocr', or 'svtrv2_mobile')
        draw_boxes: Whether to draw bounding boxes
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
        return use_tesseract(image, draw_boxes=draw_boxes)
    elif engine.lower() == 'paddleocr':
        lang = kwargs.get('lang', 'en')
        return use_paddleocr(image, lang=lang, draw_boxes=draw_boxes)
    elif engine.lower() == 'svtrv2_mobile':
        return use_svtrv2_mobile(image, draw_boxes=draw_boxes)
    else:
        raise ValueError(f"Unsupported OCR engine: {engine}. Supported engines: 'tesseract', 'paddleocr', 'svtrv2_mobile'")



def sort_boxes_reading_order(detections):
    def center_y(det):
        pts = np.array(det["points"], dtype=float)
        return np.mean(pts[:, 1])

    def left_x(det):
        pts = np.array(det["points"], dtype=float)
        return np.min(pts[:, 0])

    # Dynamic y_tolerance
    heights = [np.max(np.array(d["points"])[:, 1]) - np.min(np.array(d["points"])[:, 1]) for d in detections]
    y_tolerance = max(1.0, np.median(heights) * 0.8)

    # sort first by center_y, then left_x
    dets_sorted = sorted(detections, key=lambda d: (center_y(d), left_x(d)))

    # group into lines
    lines = []
    current_line = [dets_sorted[0]]
    current_y = center_y(dets_sorted[0])

    for det in dets_sorted[1:]:
        y = center_y(det)
        if abs(y - current_y) <= y_tolerance:
            current_line.append(det)
        else:
            lines.append(sorted(current_line, key=left_x))
            current_line = [det]
            current_y = y
    lines.append(sorted(current_line, key=left_x))

    # flatten back into reading order
    return [d for line in lines for d in line]
