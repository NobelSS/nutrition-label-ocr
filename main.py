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
from concurrent.futures import ThreadPoolExecutor, as_completed
os.makedirs("debug/preprocess", exist_ok=True) 

def normalize_text(data):
    if isinstance(data, dict):
        return json.dumps(data, sort_keys=True)  # consistent ordering
    elif data is None:
        return ""
    else:
        return str(data)

def run_pipeline(image_path: str, ocr_engine: str = 'paddleocr', draw_boxes: bool = False, ocr_lang: str = 'en') -> str:
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
                        min_area_ratio=0.1, show_steps=False, debug=False)
    
    if corners is not None:
        image = scanner.rectify_label(enhance=False)
    if image is None:
        image = cropped
    
    image = deskew(image, show_result=False, debug=False)
    image = preprocess(image, save_result=True, save_path=f'debug/preprocess/{os.path.splitext(os.path.basename(image_path))[0]}.png', debug=False)

    # compare_preprocessing_variants(deskewed)
    ocr_text = perform_ocr(image, engine=ocr_engine, draw_boxes=draw_boxes, lang=ocr_lang)

    return ocr_text


def process_single_image(image_name: str, dataset_path: str, label: dict, ocr_engine: str, draw_boxes: bool, ocr_lang: str):
    """
    Process a single image and return evaluation metrics
    
    Args:
        image_name: Name of the image file
        dataset_path: Path to dataset directory
        label: Ground truth labels dictionary
        ocr_engine: OCR engine to use
        ocr_lang: Language for OCR
        
    Returns:
        Tuple of (image_name, metric_report) or (image_name, None) if processing failed
    """
    try:
        image_path = os.path.join(dataset_path, image_name)
        output = run_pipeline(image_path, ocr_engine, draw_boxes, ocr_lang)
        
        gt = label.get(image_name, {})
        if not isinstance(gt, dict):
            return (image_name, None)
        
        if output is not None:
            parsed_output = parse_nutrition_text(output)
            metric_report = evaluate_metric(parsed_output, gt)
            return (image_name, metric_report)
        else:
            return (image_name, None)
    except Exception as e:
        print(f"Error processing {image_name}: {str(e)}")
        return (image_name, None)
    
    
def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Nutrition Label OCR Pipeline')
    parser.add_argument('--ocr-engine', choices=['tesseract', 'paddleocr', 'svtrv2_mobile'],
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
    parser.add_argument('--threads', type=int, default=4,
                       help='Number of threads for parallel processing (default: 4)')
    parser.add_argument('--draw-boxes', action='store_true', default=False,
                       help='Draw bounding boxes on the image')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()

    print(f"Using OCR engine: {args.ocr_engine}")

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
        output = run_pipeline(args.single_image, args.ocr_engine, args.draw_boxes, args.ocr_lang)

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
    
    # Get list of images to process
    all_images = os.listdir(DATASET_PATH)
    images_to_process = all_images[:args.max_images]
    
    print(f"Processing {len(images_to_process)} images using {args.threads} threads...")
    
    # Multi-threaded processing
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        # Submit all tasks
        future_to_image = {
            executor.submit(process_single_image, image, DATASET_PATH, label, args.ocr_engine, args.draw_boxes, args.ocr_lang): image
            for image in images_to_process
        }
        
        # Process completed tasks with progress bar
        for future in tqdm(as_completed(future_to_image), total=len(images_to_process), desc="Processing images"):
            image_name = future_to_image[future]
            try:
                image_name, metric_report = future.result()
                if metric_report is not None:
                    evaluation[image_name] = metric_report
            except Exception as e:
                print(f"\nError processing {image_name}: {str(e)}")

    if len(evaluation) > 0:
        evaluation_csv = f'evaluation_report_{args.ocr_engine}.csv'
        export_to_csv(evaluation, evaluation_csv)
        print(f"\nEvaluation report exported to {evaluation_csv}")
        
        # Calculate and display average metrics
        avg_field_acc = sum(e['field_accuracy'] for e in evaluation.values()) / len(evaluation)
        avg_value_acc = sum(e['value_accuracy'] for e in evaluation.values()) / len(evaluation)
        avg_unit_acc = sum(e['unit_accuracy'] for e in evaluation.values()) / len(evaluation)
        avg_percent_acc = sum(e['percent_dv_accuracy'] for e in evaluation.values()) / len(evaluation)
        
        print(f"\nAverage Metrics ({len(evaluation)} images):")
        print(f"  Field Accuracy: {avg_field_acc:.2%}")
        print(f"  Value Accuracy: {avg_value_acc:.2%}")
        print(f"  Unit Accuracy: {avg_unit_acc:.2%}")
        print(f"  Percent DV Accuracy: {avg_percent_acc:.2%}")



        

    
    
