import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from tqdm import tqdm
from object_detection import detect_object
from deskew import deskew
from perspective_correction import perspective_correction
from new_perspective_correction import NutritionLabelScanner
from preprocess import preprocess
from ocr import perform_ocr
from visuals import compare_preprocessing_variants
import json
import jiwer
import Levenshtein
import argparse
import matplotlib.pyplot as plt
from evaluation import parse_nutrition_text, evaluate, evaluate_metric, export_to_csv

def normalize_text(data):
    if isinstance(data, dict):
        return json.dumps(data, sort_keys=True)  # consistent ordering
    elif data is None:
        return ""
    else:
        return str(data)

def run_pipeline(image_path: str, ocr_engine: str = 'paddleocr', ocr_lang: str = 'en') -> str:
    """
    Run the complete OCR pipeline on an image

    Args:
        image_path: Path to the input image
        ocr_engine: OCR engine to use ('tesseract' or 'paddleocr')
        ocr_lang: Language for OCR (used with PaddleOCR)

    Returns:
        Extracted text as string
    """
    cropped = detect_object(image_path, show_result=False, draw_boxes=False)
    if cropped is None:
        print("No object detected.")
        return None

    # Option 1: Use original perspective correction
    # rectified = perspective_correction(cropped, show_result=True, debug=False)

    scanner = NutritionLabelScanner()
    corners, image = scanner.detect_label(image=cropped, canny_low=30, canny_high=100,
                        min_area_ratio=0.1, show_steps=False)
    
    if corners is not None:
        image = scanner.rectify_label(enhance=False)
    if image is None:
        image = cropped
    
    image = deskew(image, show_result=False, debug=False)
    image = preprocess(image, save_result=False, debug=False)

    # compare_preprocessing_variants(deskewed)
    ocr_text = perform_ocr(image, engine=ocr_engine, lang=ocr_lang)

    return ocr_text
    
    
def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Nutrition Label OCR Pipeline')
    parser.add_argument('--ocr-engine', choices=['tesseract', 'paddleocr'],
                       default='tesseract', help='OCR engine to use (default: tesseract)')
    parser.add_argument('--ocr-lang', default='en',
                       help='Language for OCR (default: en, used with PaddleOCR)')
    parser.add_argument('--dataset-path', default='dataset/labeled',
                       help='Path to dataset directory (default: dataset/labeled)')
    parser.add_argument('--label-path', default='dataset/label.json',
                       help='Path to label file (default: dataset/label.json)')
    parser.add_argument('--max-images', type=int, default=50,
                       help='Maximum number of images to process (default: 50)')
    parser.add_argument('--single-image', type=str,
                       help='Process a single image file instead of dataset')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()

    print(f"Using OCR engine: {args.ocr_engine}")
    if args.ocr_engine == 'paddleocr':
        print(f"OCR language: {args.ocr_lang}")

    LABEL_PATH = args.label_path
    try:
        with open(LABEL_PATH, 'r') as f:
            label = json.load(f)
            label = label.get("result", None)
    except:
        label = {}

    if label is None:
        print("No labels found.")
        exit(1)

    # Single image processing
    if args.single_image:
        if not os.path.exists(args.single_image):
            print(f"Error: Image file '{args.single_image}' not found.")
            exit(1)

        print(f'Processing single image: {args.single_image}')
        output = run_pipeline(args.single_image, args.ocr_engine, args.ocr_lang)

        if output is not None:
            print(f'OCR Output:\n{output}')
            parsed_output = parse_nutrition_text(output)
            print(f'Parsed Output:\n{parsed_output}')
            evaluation_report = evaluate(parsed_output, label.get(os.path.basename(args.single_image), {}))
            print(f'Evaluation Report:\n{json.dumps(evaluation_report, indent=2)}')
            metric_report = evaluate_metric(parsed_output, label.get(os.path.basename(args.single_image), {}))
            print(f'Metric Report:\n{json.dumps(metric_report, indent=2)}')
        else:
            print("OCR failed or no object detected.")
        exit(0)

    # Dataset processing
    DATASET_PATH = args.dataset_path

    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset path '{DATASET_PATH}' not found.")
        exit(1)

    evaluation = {}

    for idx, image in enumerate(tqdm(os.listdir(DATASET_PATH), desc="Processing images")):
        output = run_pipeline(os.path.join(DATASET_PATH, image), args.ocr_engine, args.ocr_lang)
        gt = label.get(image, "N/A")
        print(f"\nProcessing: {image}")

        if output is not None:
            parsed_output = parse_nutrition_text(output)
            metric_report = evaluate_metric(parsed_output, gt)
            evaluation[image] = metric_report
        else:
            print("No object detected on cropping.\n")

        if idx >= args.max_images - 1:
            break

    if len(evaluation) > 0:
        evaluation_csv = 'evaluation_report_sharp_mask.csv'
        export_to_csv(evaluation, evaluation_csv)
        print(f"Evaluation report exported to {evaluation_csv}")



        

    
    
