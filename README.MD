# Nutrition Label Detector

A computer vision pipeline for detecting and extracting text from nutrition labels using OCR.

## Features

- Object detection for nutrition labels
- Perspective correction and image preprocessing
- Multiple OCR engines support:
  - **Tesseract OCR** (default)
  - **PaddleOCR** (alternative option)
- Evaluation metrics (CER, WER)

## Setup Instructions

### 1. Create Virtual Environment

```bash
python3 -m venv venv
venv/Scripts/activate  # On Windows
# source venv/bin/activate  # On Linux/Mac
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Copy `.env.example` to `.env` and update it with your environment-specific values.

## Usage

### Basic Usage (Default Tesseract)

```bash
python main.py
```

### Using PaddleOCR

```bash
python main.py --ocr-engine paddleocr
```

### Process Single Image

```bash
# With Tesseract
python main.py --single-image path/to/image.jpg

# With PaddleOCR
python main.py --single-image path/to/image.jpg --ocr-engine paddleocr
```

### Command Line Options

```bash
python main.py --help
```

Available options:

- `--ocr-engine`: Choose OCR engine (`tesseract` or `paddleocr`)
- `--ocr-lang`: Language for OCR (default: `en`, used with PaddleOCR)
- `--dataset-path`: Path to dataset directory
- `--label-path`: Path to label file
- `--max-images`: Maximum number of images to process
- `--single-image`: Process a single image file

### Testing OCR Engines

Compare the performance of different OCR engines:

```bash
python test_ocr_engines.py path/to/test/image.jpg
```

## OCR Engines

### Tesseract OCR

- Default OCR engine
- Requires Tesseract to be installed on the system
- Good for general text recognition

### PaddleOCR

- Alternative OCR engine by PaddlePaddle
- No external dependencies (self-contained)
- Often better for complex layouts and multilingual text
- Supports multiple languages

## Dependencies

Main dependencies include:

- OpenCV for image processing
- PyTorch for object detection
- Tesseract OCR
- PaddleOCR (optional)
- scikit-image for image transformations
