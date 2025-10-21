import cv2
import pytesseract
import json
from openocr import OpenOCR  # or your OpenOCR inference wrapper
from PIL import Image
import numpy as np

# ============================
# 1️⃣ TESSERACT DETECTION
# ============================

def detect_with_tesseract(image_path, output_path="tesseract_boxes.jpg"):
    """
    Detect text regions using Tesseract and draw bounding boxes.
    """
    img = cv2.imread(image_path)
    h, w, _ = img.shape

    # Run Tesseract OCR with detailed box info
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

    for i in range(len(data["text"])):
        if int(data["conf"][i]) > 0 and data["text"][i].strip() != "":
            (x, y, bw, bh) = (data["left"][i], data["top"][i], data["width"][i], data["height"][i])
            cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.putText(img, data["text"][i], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imwrite(output_path, img)
    print(f"[Tesseract] Detection boxes saved to: {output_path}")
    return img


# ============================
# 2️⃣ OPENOCR (DBNet) DETECTION
# ============================

def detect_with_openocr(image_path, output_path="openocr_boxes.jpg"):
    """
    Detect text regions using OpenOCR (DBNet) and draw bounding boxes.
    """
    # Initialize OpenOCR (make sure weights are set up)
    ocr = OpenOCR(backend='onnx', device='cpu', onnx_rec_model_path='./model/rec/repsvtr_ch.onnx')

    result = ocr(image_path)  # returns list of boxes
    img = cv2.imread(image_path)

    # OpenOCR returns a tuple: (results_list, timing_info)
    if isinstance(result, tuple) and len(result) >= 1:
        results_data = result[0]
        if isinstance(results_data, list) and len(results_data) > 0:
            # Check if it's a string with JSON data (file path input)
            if isinstance(results_data[0], str) and '\t' in results_data[0]:
                # Extract the JSON string from the first element
                json_string = results_data[0].split('\t')[1]  # Split by tab and take the JSON part
                detections = json.loads(json_string)
            # Check if it's a list of detections (numpy array input)
            elif isinstance(results_data[0], list):
                detections = results_data[0]
            else:
                print("[WARN] No valid results data in tuple.")
                return img
        else:
            print("[WARN] No valid results data in tuple.")
            return img
    elif isinstance(result, list):
        detections = result
    elif isinstance(result, dict) and "result" in result:
        detections = result["result"]
    else:
        print("[WARN] No valid detections returned.")
        return img

    for det in detections:
        if not isinstance(det, dict):
            continue

        points = det.get("points", [])
        text = det.get("transcription", "")
        score = det.get("score", 0)

        # --- Validate and normalize points ---
        if not points or not isinstance(points[0], (list, tuple)):
            continue

        pts = np.array(points, dtype=np.int32)

        # Draw polygon (red)
        cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

        # Label (blue)
        x, y = pts[0]
        cv2.putText(
            img,
            f"{text} ({score:.2f})",
            (x, max(20, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )
        
    cv2.imwrite(output_path, img)
    print(f"[OpenOCR] Detection boxes saved to: {output_path}")
    return img


# ============================
# ✅ Example usage
# ============================

if __name__ == "__main__":
    image_path = "dataset/labeled/image_57.png"

    detect_with_tesseract(image_path)
    detect_with_openocr(image_path)
