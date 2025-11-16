"""
Experimental preprocessing pipeline for testing different combinations
of detection and OCR preprocessing methods.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import cv2
import json
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from new_perspective_correction import NutritionLabelScanner
from preprocess import (
    histogram_equalization,
    enhance_contrast,
    binarize,
    denoise,
    sharpen_mask,
    sharpen_kernel
)
from evaluation import parse_nutrition_text, evaluate_metric, export_to_csv


# ============================
# DETECTION PREPROCESSING METHODS
# ============================

def detection_clahe_lab_before_grayscale(image: np.ndarray, use_bilateral: bool = False):
    """
    Detection preprocessing: CLAHE on LAB color space before grayscale conversion.
    This is the existing method from NutritionLabelScanner.enhance_contrast
    """
    scanner = NutritionLabelScanner()
    enhanced = scanner.enhance_contrast(image)  # CLAHE on LAB before grayscale
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    
    if use_bilateral:
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
    
    return gray


def detection_clahe_after_grayscale(image: np.ndarray, use_bilateral: bool = False):
    """
    Detection preprocessing: CLAHE applied after converting to grayscale.
    Uses enhance_contrast_grayscale from NutritionLabelScanner
    """
    scanner = NutritionLabelScanner()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    if use_bilateral:
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
    
    enhanced = scanner.enhance_contrast_grayscale(gray)  # CLAHE after grayscale
    return enhanced


# ============================
# OCR PREPROCESSING METHODS
# ============================

def preprocess_no_filter(image: np.ndarray):
    """Preprocess: No filter - return image as is, then sharpen_mask"""
    image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    return gray

def preprocess_sharpen_mask(image: np.ndarray):
    """Preprocess: Sharpen mask"""
    image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    return sharpen_mask(gray)

def preprocess_sharpen_kernel(image: np.ndarray):
    """Preprocess: Sharpen kernel"""
    image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    return sharpen_kernel(gray)

def preprocess_gaussian_blur(image: np.ndarray):
    """Preprocess: Gaussian Blur + sharpen_mask"""
    image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    return sharpen_mask(blurred)


def preprocess_bilateral_filter(image: np.ndarray):
    """Preprocess: Bilateral Filter + sharpen_mask"""
    image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    return sharpen_mask(filtered)


def preprocess_clahe_gaussian_blur(image: np.ndarray):
    """Preprocess: CLAHE + Gaussian Blur + sharpen_mask"""
    image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    enhanced = enhance_contrast(gray)  # CLAHE
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    return sharpen_mask(blurred)


def preprocess_histogram_eq_gaussian_blur(image: np.ndarray):
    """Preprocess: Histogram Equalization + Gaussian Blur + sharpen_mask"""
    image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    equalized = histogram_equalization(gray)
    blurred = cv2.GaussianBlur(equalized, (3, 3), 0)
    return sharpen_mask(blurred)

def preprocess_clahe_gaussian_blur_binarization(image: np.ndarray):
    """Preprocess: CLAHE + Gaussian Blur + sharpen_mask + Binarization"""
    image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    enhanced = enhance_contrast(gray)  # CLAHE
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    sharpened = sharpen_mask(blurred)
    return binarize(sharpened)


def preprocess_histogram_eq_gaussian_blur_binarization(image: np.ndarray):
    """Preprocess: Histogram Equalization + Gaussian Blur + sharpen_mask + Binarization"""
    image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    equalized = histogram_equalization(gray)
    blurred = cv2.GaussianBlur(equalized, (3, 3), 0)
    sharpened = sharpen_mask(blurred)
    return binarize(sharpened)


# ============================
# EXPERIMENT CONFIGURATIONS
# ============================

# Define all experiment configurations
EXPERIMENT_CONFIGS = {
    'a': {
        'name': 'Preprocess No Filter (Detection CLAHE after grayscale)',
        'detection_method': detection_clahe_after_grayscale,
        'detection_params': {'use_bilateral': False},
        'preprocess_method': preprocess_no_filter
    },
    'b': {
        'name': 'Preprocess No Filter (Detection CLAHE LAB before grayscale)',
        'detection_method': detection_clahe_lab_before_grayscale,
        'detection_params': {'use_bilateral': False},
        'preprocess_method': preprocess_no_filter
    },
    'c': {
        'name': 'Preprocess No Filter (Detection CLAHE LAB before grayscale + Bilateral Filter)',
        'detection_method': detection_clahe_lab_before_grayscale,
        'detection_params': {'use_bilateral': True},
        'preprocess_method': preprocess_no_filter
    },
    'd': {
        'name': 'Preprocess Sharpen Mask (Detection CLAHE after grayscale)',
        'detection_method': detection_clahe_after_grayscale,
        'detection_params': {'use_bilateral': False},
        'preprocess_method': preprocess_sharpen_mask
    },
    'e': {
        'name': 'Preprocess Sharpen Kernel (Detection CLAHE after grayscale)',
        'detection_method': detection_clahe_after_grayscale,
        'detection_params': {'use_bilateral': False},
        'preprocess_method': preprocess_sharpen_kernel
    },
    'f': {
        'name': 'Preprocess + GaussianBlur + Sharpen Mask (Detection CLAHE LAB before grayscale + Bilateral Filter)',
        'detection_method': detection_clahe_lab_before_grayscale,
        'detection_params': {'use_bilateral': True},
        'preprocess_method': preprocess_gaussian_blur
    },
    'g': {
        'name': 'Preprocess + Bilateral Filter + Sharpen Mask (Detection CLAHE LAB before grayscale + Bilateral Filter)',
        'detection_method': detection_clahe_lab_before_grayscale,
        'detection_params': {'use_bilateral': True},
        'preprocess_method': preprocess_bilateral_filter
    },
    'h': {
        'name': 'Preprocess + CLAHE + GaussianBlur + Sharpen Mask + Binarization (Detection CLAHE LAB before grayscale + Bilateral Filter)',
        'detection_method': detection_clahe_lab_before_grayscale,
        'detection_params': {'use_bilateral': True},
        'preprocess_method': preprocess_clahe_gaussian_blur_binarization
    },
    'i': {
        'name': 'Preprocess + Histogram Equalization + GaussianBlur + Sharpen Mask + Binarization (Detection CLAHE LAB before grayscale + Bilateral Filter)',
        'detection_method': detection_clahe_lab_before_grayscale,
        'detection_params': {'use_bilateral': True},
        'preprocess_method': preprocess_histogram_eq_gaussian_blur_binarization
    }
}


# ============================
# EXPERIMENTAL PIPELINE RUNNER
# ============================

def run_experimental_pipeline(image_path: str, experiment_id: str, 
                              ocr_engine: str = 'paddleocr', 
                              draw_boxes: bool = False,
                              ocr_lang: str = 'en',
                              show_steps: bool = False,
                              debug: bool = False):
    """
    Run the experimental pipeline with specified configuration.
    
    Args:
        image_path: Path to input image
        experiment_id: One of 'a' through 'i' corresponding to experiment configs
        ocr_engine: OCR engine to use ('tesseract', 'paddleocr', or 'svtrv2_mobile')
        draw_boxes: Whether to draw bounding boxes
        ocr_lang: Language for OCR
        show_steps: Whether to show intermediate steps
        debug: Whether to print debug information
        
    Returns:
        Extracted text as string, or None if processing failed
    """
    if experiment_id not in EXPERIMENT_CONFIGS:
        raise ValueError(f"Invalid experiment_id: {experiment_id}. Must be one of {list(EXPERIMENT_CONFIGS.keys())}")
    
    config = EXPERIMENT_CONFIGS[experiment_id]
    
    if debug:
        print(f"Running experiment: {config['name']}")
    
    # Import here to avoid circular imports
    from object_detection import detect_object
    from deskew import deskew
    from new_perspective_correction import NutritionLabelScanner
    from ocr import perform_ocr
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return None
    
    # Step 1: Object Detection
    cropped = detect_object(image_path, show_result=False, draw_boxes=False)
    if cropped is None:
        if debug:
            print("No object detected.")
        return None
    
    # Step 2: Perspective Correction with custom detection preprocessing
    scanner = NutritionLabelScanner()
    
    # Modify the scanner's preprocess_for_labels to use our custom detection method
    original_preprocess = scanner.preprocess_for_labels
    
    def custom_preprocess_for_labels(image, blur_kernel=3, canny_low=30, canny_high=100):
        """Custom preprocessing that uses our experimental detection method"""
        resized = scanner.resize_image(image)
        
        # Apply custom detection preprocessing
        gray = config['detection_method'](resized, **config['detection_params'])
        
        # Continue with edge detection
        blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
        edges = cv2.Canny(blurred, canny_low, canny_high)
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        return edges, resized
    
    scanner.preprocess_for_labels = custom_preprocess_for_labels
    
    try:
        corners, image = scanner.detect_label(
            image=cropped, 
            canny_low=30, 
            canny_high=100,
            min_area_ratio=0.1, 
            show_steps=show_steps, 
            debug=debug
        )
        
        if corners is not None:
            image = scanner.rectify_label()
        if image is None:
            image = cropped
    finally:
        # Restore original method
        scanner.preprocess_for_labels = original_preprocess
    
    # Step 3: Deskew
    image = deskew(image, show_result=False, debug=debug)
    
    # Step 4: OCR Preprocessing (experimental)
    image = config['preprocess_method'](image)
    
    # Step 5: OCR
    ocr_text = perform_ocr(image, engine=ocr_engine, draw_boxes=draw_boxes, lang=ocr_lang)
    
    return ocr_text


# ============================
# DATASET EVALUATION (reusing existing pattern)
# ============================

def process_single_image(image_name: str, dataset_path: str, label: dict, 
                         ocr_engine: str, draw_boxes: bool, ocr_lang: str, 
                         experiment_id: str):
    """
    Process a single image with specified experiment configuration and return evaluation metrics.
    Reuses the same pattern as main.py's process_single_image.
    
    Args:
        image_name: Name of the image file
        dataset_path: Path to dataset directory
        label: Ground truth labels dictionary
        ocr_engine: OCR engine to use
        draw_boxes: Whether to draw bounding boxes
        ocr_lang: Language for OCR
        experiment_id: Experiment ID (a-i)
        
    Returns:
        Tuple of (image_name, metric_report) or (image_name, None) if processing failed
    """
    try:
        image_path = os.path.join(dataset_path, image_name)
        output = run_experimental_pipeline(
            image_path, 
            experiment_id, 
            ocr_engine, 
            draw_boxes, 
            ocr_lang,
            show_steps=False,
            debug=False
        )
        
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


def run_dataset_evaluation(dataset_path: str, label: dict, args, experiment_id: str):
    """
    Run evaluation on dataset with specified experiment configuration.
    Reuses the same pattern as main.py's run_dataset_evaluation.
    """
    evaluation = {}
    
    # Get list of images to process
    all_images = [f for f in os.listdir(dataset_path) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    images_to_process = all_images[:args.max_images]
    
    config = EXPERIMENT_CONFIGS[experiment_id]
    print(f"Processing {len(images_to_process)} images using {args.threads} threads with experiment '{experiment_id}': {config['name']}...")
    
    # Multi-threaded processing
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        # Submit all tasks
        future_to_image = {
            executor.submit(process_single_image, image, dataset_path, label, 
                          args.ocr_engine, args.draw_boxes, args.ocr_lang, experiment_id): image
            for image in images_to_process
        }
        
        # Process completed tasks with progress bar
        for future in tqdm(as_completed(future_to_image), total=len(images_to_process), 
                         desc=f"Experiment {experiment_id}"):
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
        
        # Create experimental directory
        exp_dir = f'{ocr_dir}/experimental'
        os.makedirs(exp_dir, exist_ok=True)
        
        evaluation_csv = f'{exp_dir}/evaluation_report_experiment_{experiment_id}.csv'
        export_to_csv(evaluation, evaluation_csv)
        print(f"\nEvaluation report exported to {evaluation_csv}")
        
        # Calculate and display average metrics (same as main.py)
        avg_field_acc = sum(e['field_accuracy'] for e in evaluation.values()) / len(evaluation)
        avg_value_acc = sum(e['value_accuracy'] for e in evaluation.values()) / len(evaluation)
        avg_unit_acc = sum(e['unit_accuracy'] for e in evaluation.values()) / len(evaluation)
        avg_percent_acc = sum(e['percent_dv_accuracy'] for e in evaluation.values()) / len(evaluation)
        
        print(f"\nAverage Metrics for experiment '{experiment_id}' ({len(evaluation)} images):")
        print(f"  Field Accuracy: {avg_field_acc:.2%}")
        print(f"  Value Accuracy: {avg_value_acc:.2%}")
        print(f"  Unit Accuracy: {avg_unit_acc:.2%}")
        print(f"  Percent DV Accuracy: {avg_percent_acc:.2%}")
        
        return {
            'experiment_id': experiment_id,
            'name': config['name'],
            'field_accuracy': avg_field_acc,
            'value_accuracy': avg_value_acc,
            'unit_accuracy': avg_unit_acc,
            'percent_dv_accuracy': avg_percent_acc,
            'total_images': len(evaluation)
        }
    
    return None


def list_experiments():
    """List all available experiment configurations"""
    print("Available Experiments:")
    print("=" * 80)
    for exp_id, config in EXPERIMENT_CONFIGS.items():
        print(f"{exp_id}. {config['name']}")
    print("=" * 80)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Experimental Preprocessing Pipeline')
    parser.add_argument('--image', type=str, help='Path to input image (for single image mode)')
    parser.add_argument('--experiment', type=str, choices=list(EXPERIMENT_CONFIGS.keys()),
                       help='Experiment ID (a-i) - required for single image or single experiment mode')
    parser.add_argument('--ocr-engine', choices=['tesseract', 'svtrv2_mobile'],
                       default='tesseract', help='OCR engine to use (default: tesseract)')
    parser.add_argument('--ocr-lang', default='en',
                       help='Language for OCR (default: en, used with PaddleOCR)')
    parser.add_argument('--dataset-path', default='dataset/filtered',
                       help='Path to dataset directory (default: dataset/filtered)')
    parser.add_argument('--label-path', default='dataset/label.json',
                       help='Path to label file (default: dataset/label.json)')
    parser.add_argument('--max-images', type=int, default=200,
                       help='Maximum number of images to process (default: 200)')
    parser.add_argument('--threads', type=int, default=4,
                       help='Number of threads for parallel processing (default: 4)')
    parser.add_argument('--draw-boxes', action='store_true', default=False,
                       help='Draw bounding boxes on the image')
    parser.add_argument('--show-steps', action='store_true', default=False,
                       help='Show intermediate steps')
    parser.add_argument('--debug', action='store_true', default=False,
                       help='Print debug information')
    parser.add_argument('--list', action='store_true', default=False,
                       help='List all available experiments')
    parser.add_argument('--test-all-experiments', action='store_true', default=False,
                       help='Test all experiments (a-i) sequentially on dataset')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    print(f"Using OCR engine: {args.ocr_engine}")

    # Load labels
    LABEL_PATH = args.label_path
    try:
        with open(LABEL_PATH, 'r') as f:
            label = json.load(f)
            label = label.get("result", None)
    except:
        label = {}

    if label is None:
        label = {}

    # List experiments
    if args.list:
        list_experiments()
        exit(0)

    # Single image processing
    if args.image:
        if not os.path.exists(args.image):
            print(f"Error: Image file '{args.image}' not found.")
            exit(1)
        
        if args.experiment is None:
            print("Error: --experiment is required when using --image")
            exit(1)

        print(f'Processing single image: {args.image} with experiment {args.experiment}')
        output = run_experimental_pipeline(
            image_path=args.image,
            experiment_id=args.experiment,
            ocr_engine=args.ocr_engine,
            draw_boxes=args.draw_boxes,
            ocr_lang=args.ocr_lang,
            show_steps=args.show_steps,
            debug=args.debug
        )
        
        if output:
            print(f'\nOCR Output:\n{output}')
            parsed_output = parse_nutrition_text(output)
            print(f'\nParsed Output:\n{parsed_output}')
            
            image_name = os.path.basename(args.image)
            gt = label.get(image_name, {})
            if gt:
                from evaluation import evaluate
                evaluation_report = evaluate(parsed_output, gt)
                print(f'\nEvaluation Report:\n{json.dumps(evaluation_report, indent=2)}')
                metric_report = evaluate_metric(parsed_output, gt)
                print(f'\nMetric Report:\n{json.dumps(metric_report, indent=2)}')
        else:
            print("OCR failed or no object detected.")
        exit(0)

    # Dataset processing
    DATASET_PATH = args.dataset_path

    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset path '{DATASET_PATH}' not found.")
        exit(1)

    # Test all experiments if requested
    if args.test_all_experiments:
        print("Testing all experiments...")
        all_results = []
        
        for experiment_id in sorted(EXPERIMENT_CONFIGS.keys()):
            print(f"\n{'='*80}")
            print(f"Testing experiment '{experiment_id}': {EXPERIMENT_CONFIGS[experiment_id]['name']}")
            print(f"{'='*80}")
            
            result = run_dataset_evaluation(DATASET_PATH, label, args, experiment_id)
            if result:
                all_results.append(result)
        
        # Print summary of all experiments (same format as main.py)
        if all_results:
            print(f"\n{'='*80}")
            print("SUMMARY OF ALL EXPERIMENTS")
            print(f"{'='*80}")
            print(f"{'Exp':<5} {'Field Acc':<12} {'Value Acc':<12} {'Unit Acc':<12} {'Percent DV':<12} {'Images':<8}")
            print("-" * 80)
            for result in all_results:
                print(f"{result['experiment_id']:<5} {result['field_accuracy']:<12.2%} {result['value_accuracy']:<12.2%} "
                      f"{result['unit_accuracy']:<12.2%} {result['percent_dv_accuracy']:<12.2%} {result['total_images']:<8}")
            
            # Find best performing experiment for each metric
            print(f"\n{'='*80}")
            print("BEST PERFORMING EXPERIMENTS")
            print(f"{'='*80}")
            best_field = max(all_results, key=lambda x: x['field_accuracy'])
            best_value = max(all_results, key=lambda x: x['value_accuracy'])
            best_unit = max(all_results, key=lambda x: x['unit_accuracy'])
            best_percent = max(all_results, key=lambda x: x['percent_dv_accuracy'])
            
            print(f"Best Field Accuracy:     {best_field['experiment_id']} ({best_field['field_accuracy']:.2%}) - {best_field['name']}")
            print(f"Best Value Accuracy:     {best_value['experiment_id']} ({best_value['value_accuracy']:.2%}) - {best_value['name']}")
            print(f"Best Unit Accuracy:      {best_unit['experiment_id']} ({best_unit['unit_accuracy']:.2%}) - {best_unit['name']}")
            print(f"Best Percent DV Accuracy: {best_percent['experiment_id']} ({best_percent['percent_dv_accuracy']:.2%}) - {best_percent['name']}")
    else:
        # Single experiment processing
        if args.experiment is None:
            print("Error: --experiment is required for dataset processing, or use --test-all-experiments")
            exit(1)
        
        result = run_dataset_evaluation(DATASET_PATH, label, args, args.experiment)
