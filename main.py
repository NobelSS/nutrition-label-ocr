import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
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
from constants import PipelineVariation
os.makedirs("debug/preprocess", exist_ok=True)

def normalize_text(data):
    if isinstance(data, dict):
        return json.dumps(data, sort_keys=True)  # consistent ordering
    elif data is None:
        return ""
    else:
        return str(data)

def run_pipeline(image_path: str, ocr_engine: str = 'paddleocr', draw_boxes: bool = False, 
                ocr_lang: str = 'en', variation: PipelineVariation = PipelineVariation.FULL_PIPELINE) -> str:
    """
    Run the OCR pipeline on an image with specified variation

    Args:
        image_path: Path to the input image
        ocr_engine: OCR engine to use ('tesseract' or 'paddleocr')
        draw_boxes: Whether to draw bounding boxes
        ocr_lang: Language for OCR (used with PaddleOCR)
        variation: Pipeline variation to use

    Returns:
        Extracted text as string
    """
    image = cv2.imread(image_path)
    
    # No pipeline - straight OCR
    if variation == PipelineVariation.NO_PIPELINE:
        ocr_text = perform_ocr(image, engine=ocr_engine, draw_boxes=draw_boxes, lang=ocr_lang)
        return ocr_text
    
    # Full pipeline
    if variation == PipelineVariation.FULL_PIPELINE:
        cropped = detect_object(image_path, show_result=False, draw_boxes=False)
        if cropped is None:
            print("No object detected.")
            return None

        scanner = NutritionLabelScanner()
        corners, image = scanner.detect_label(image=cropped, canny_low=30, canny_high=100,
                            min_area_ratio=0.1, show_steps=False, debug=False)
        
        if corners is not None:
            image = scanner.rectify_label(enhance=False)
        if image is None:
            image = cropped
        
        image = deskew(image, show_result=False, debug=False)
        image = preprocess(image, save_result=True, save_path=f'debug/preprocess/{os.path.splitext(os.path.basename(image_path))[0]}.png', debug=False)
        
        ocr_text = perform_ocr(image, engine=ocr_engine, draw_boxes=draw_boxes, lang=ocr_lang)
        return ocr_text
    
    # No object detection - skip object detection but keep rest
    if variation == PipelineVariation.NO_OBJECT_DETECTION:
        scanner = NutritionLabelScanner()
        corners, image = scanner.detect_label(image=image, canny_low=30, canny_high=100,
                            min_area_ratio=0.1, show_steps=False, debug=False)
        
        if corners is not None:
            image = scanner.rectify_label(enhance=False)
        
        image = deskew(image, show_result=False, debug=False)
        image = preprocess(image, save_result=True, save_path=f'debug/preprocess/{os.path.splitext(os.path.basename(image_path))[0]}.png', debug=False)
        
        ocr_text = perform_ocr(image, engine=ocr_engine, draw_boxes=draw_boxes, lang=ocr_lang)
        return ocr_text
    
    # No rectification - skip perspective correction but keep rest
    if variation == PipelineVariation.NO_RECTIFICATION:
        cropped = detect_object(image_path, show_result=False, draw_boxes=False)
        if cropped is None:
            print("No object detected.")
            return None

        # Skip perspective correction (NutritionLabelScanner)
        image = cropped
        
        image = deskew(image, show_result=False, debug=False)
        image = preprocess(image, save_result=True, save_path=f'debug/preprocess/{os.path.splitext(os.path.basename(image_path))[0]}.png', debug=False)
        
        ocr_text = perform_ocr(image, engine=ocr_engine, draw_boxes=draw_boxes, lang=ocr_lang)
        return ocr_text
    
    # No deskew - skip deskew but keep rest
    if variation == PipelineVariation.NO_DESKEW:
        cropped = detect_object(image_path, show_result=False, draw_boxes=False)
        if cropped is None:
            print("No object detected.")
            return None

        scanner = NutritionLabelScanner()
        corners, image = scanner.detect_label(image=cropped, canny_low=30, canny_high=100,
                            min_area_ratio=0.1, show_steps=False, debug=False)
        
        if corners is not None:
            image = scanner.rectify_label(enhance=False)
        if image is None:
            image = cropped
        
        # Skip deskew
        image = preprocess(image, save_result=True, save_path=f'debug/preprocess/{os.path.splitext(os.path.basename(image_path))[0]}.png', debug=False)
        
        ocr_text = perform_ocr(image, engine=ocr_engine, draw_boxes=draw_boxes, lang=ocr_lang)
        return ocr_text
    
    # No deskew and no rectification - skip both deskew and perspective correction
    if variation == PipelineVariation.NO_DESKEW_NO_RECTIFICATION:
        cropped = detect_object(image_path, show_result=False, draw_boxes=False)
        if cropped is None:
            print("No object detected.")
            return None

        # Skip perspective correction (NutritionLabelScanner) and deskew
        image = cropped
        
        # Skip deskew
        image = preprocess(image, save_result=True, save_path=f'debug/preprocess/{os.path.splitext(os.path.basename(image_path))[0]}.png', debug=False)
        
        ocr_text = perform_ocr(image, engine=ocr_engine, draw_boxes=draw_boxes, lang=ocr_lang)
        return ocr_text
    
    # No preprocess - skip preprocess but keep rest
    if variation == PipelineVariation.NO_PREPROCESS:
        cropped = detect_object(image_path, show_result=False, draw_boxes=False)
        if cropped is None:
            print("No object detected.")
            return None

        scanner = NutritionLabelScanner()
        corners, image = scanner.detect_label(image=cropped, canny_low=30, canny_high=100,
                            min_area_ratio=0.1, show_steps=False, debug=False)
        
        if corners is not None:
            image = scanner.rectify_label(enhance=False)
        if image is None:
            image = cropped
        
        image = deskew(image, show_result=False, debug=False)
        # Skip preprocess
        
        ocr_text = perform_ocr(image, engine=ocr_engine, draw_boxes=draw_boxes, lang=ocr_lang)
        return ocr_text
    
    return None


def process_single_image(image_name: str, dataset_path: str, label: dict, ocr_engine: str, 
                        draw_boxes: bool, ocr_lang: str, variation: PipelineVariation):
    """
    Process a single image and return evaluation metrics
    
    Args:
        image_name: Name of the image file
        dataset_path: Path to dataset directory
        label: Ground truth labels dictionary
        ocr_engine: OCR engine to use
        draw_boxes: Whether to draw bounding boxes
        ocr_lang: Language for OCR
        variation: Pipeline variation to use
        
    Returns:
        Tuple of (image_name, metric_report) or (image_name, None) if processing failed
    """
    try:
        image_path = os.path.join(dataset_path, image_name)
        output = run_pipeline(image_path, ocr_engine, draw_boxes, ocr_lang, variation)
        
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
    parser.add_argument('--pipeline', choices=[v.value for v in PipelineVariation],
                       default=PipelineVariation.FULL_PIPELINE.value,
                       help='Pipeline variation to use (default: full_pipeline)')
    parser.add_argument('--test-all-variations', action='store_true', default=False,
                       help='Test all pipeline variations sequentially')

    return parser.parse_args()

def run_dataset_evaluation(dataset_path: str, label: dict, args, variation: PipelineVariation):
    """Run evaluation on dataset with specified variation"""
    evaluation = {}
    
    # Get list of images to process
    all_images = os.listdir(dataset_path)
    images_to_process = all_images[:args.max_images]
    
    print(f"Processing {len(images_to_process)} images using {args.threads} threads with {variation.value} variation...")
    
    # Multi-threaded processing
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        # Submit all tasks
        future_to_image = {
            executor.submit(process_single_image, image, dataset_path, label, args.ocr_engine, args.draw_boxes, args.ocr_lang, variation): image
            for image in images_to_process
        }
        
        # Process completed tasks with progress bar
        for future in tqdm(as_completed(future_to_image), total=len(images_to_process), desc=f"Processing {variation.value}"):
            image_name = future_to_image[future]
            try:
                image_name, metric_report = future.result()
                if metric_report is not None:
                    evaluation[image_name] = metric_report
            except Exception as e:
                print(f"\nError processing {image_name}: {str(e)}")

    if len(evaluation) > 0:
        # Create OCR engine specific directory
        ocr_dir = f'evaluation/{args.ocr_engine}'
        os.makedirs(ocr_dir, exist_ok=True)
        
        evaluation_csv = f'{ocr_dir}/evaluation_report_{variation.value}.csv'
        export_to_csv(evaluation, evaluation_csv)
        print(f"\nEvaluation report exported to {evaluation_csv}")
        
        # Calculate and display average metrics
        avg_field_acc = sum(e['field_accuracy'] for e in evaluation.values()) / len(evaluation)
        avg_value_acc = sum(e['value_accuracy'] for e in evaluation.values()) / len(evaluation)
        avg_unit_acc = sum(e['unit_accuracy'] for e in evaluation.values()) / len(evaluation)
        avg_percent_acc = sum(e['percent_dv_accuracy'] for e in evaluation.values()) / len(evaluation)
        
        print(f"\nAverage Metrics for {variation.value} ({len(evaluation)} images):")
        print(f"  Field Accuracy: {avg_field_acc:.2%}")
        print(f"  Value Accuracy: {avg_value_acc:.2%}")
        print(f"  Unit Accuracy: {avg_unit_acc:.2%}")
        print(f"  Percent DV Accuracy: {avg_percent_acc:.2%}")
        
        return {
            'variation': variation.value,
            'field_accuracy': avg_field_acc,
            'value_accuracy': avg_value_acc,
            'unit_accuracy': avg_unit_acc,
            'percent_dv_accuracy': avg_percent_acc,
            'total_images': len(evaluation)
        }
    
    return None

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

        variation = PipelineVariation(args.pipeline)
        print(f'Processing single image: {args.single_image} with {variation.value} variation')
        output = run_pipeline(args.single_image, args.ocr_engine, args.draw_boxes, args.ocr_lang, variation)

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

    # Test all variations if requested
    if args.test_all_variations:
        print("Testing all pipeline variations...")
        all_results = []
        
        for variation in PipelineVariation:
            print(f"\n{'='*50}")
            print(f"Testing {variation.value} variation")
            print(f"{'='*50}")
            
            result = run_dataset_evaluation(DATASET_PATH, label, args, variation)
            if result:
                all_results.append(result)
        
        # Print summary of all variations
        if all_results:
            print(f"\n{'='*50}")
            print("SUMMARY OF ALL VARIATIONS")
            print(f"{'='*50}")
            print(f"{'Variation':<20} {'Field Acc':<10} {'Value Acc':<10} {'Unit Acc':<10} {'Percent DV':<10}")
            print("-" * 70)
            for result in all_results:
                print(f"{result['variation']:<20} {result['field_accuracy']:<10.2%} {result['value_accuracy']:<10.2%} {result['unit_accuracy']:<10.2%} {result['percent_dv_accuracy']:<10.2%}")
    else:
        # Single variation processing
        variation = PipelineVariation(args.pipeline)
        result = run_dataset_evaluation(DATASET_PATH, label, args, variation)
