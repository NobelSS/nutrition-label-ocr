"""
Script to add random rotation to images in dataset/labeled
Rotates images between -10 and 10 degrees to mimic real-world picture angles
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
import random

def rotate_image(image, angle):
    """
    Rotate image by given angle in degrees.
    Expands the canvas to fit the rotated image (with black borders).
    This mimics real-world photos where the full image is visible, just rotated.
    
    Args:
        image: Input image (numpy array)
        angle: Rotation angle in degrees (positive = counter-clockwise)
    
    Returns:
        Rotated image with expanded canvas
    """
    # Get image dimensions
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new dimensions to fit rotated image
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # Adjust rotation matrix for new center
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]
    
    # Rotate image with expanded canvas (black borders)
    rotated = cv2.warpAffine(image, rotation_matrix, (new_w, new_h), 
                             flags=cv2.INTER_LINEAR, 
                             borderMode=cv2.BORDER_CONSTANT, 
                             borderValue=(0, 0, 0))
    
    return rotated

def process_dataset(source_dir='dataset/labeled', output_dir='dataset/rotated', 
                   min_angle=-10, max_angle=10, seed=None):
    """
    Process all images in source directory, apply random rotation, and save to output directory.
    
    Args:
        source_dir: Source directory containing images
        output_dir: Output directory for rotated images
        min_angle: Minimum rotation angle in degrees
        max_angle: Maximum rotation angle in degrees
        seed: Random seed for reproducibility (None for random)
    """
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    for filename in os.listdir(source_dir):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_files.append(filename)
    
    if not image_files:
        print(f"No image files found in {source_dir}")
        return
    
    print(f"Found {len(image_files)} images to process")
    print(f"Rotation range: {min_angle}° to {max_angle}°")
    
    # Process each image
    for filename in tqdm(image_files, desc="Rotating images"):
        source_path = os.path.join(source_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        # Read image
        image = cv2.imread(source_path)
        if image is None:
            print(f"Warning: Could not read {filename}, skipping...")
            continue
        
        # Generate random rotation angle
        angle = random.uniform(min_angle, max_angle)
        
        # Rotate image
        rotated_image = rotate_image(image, angle)
        
        # Save rotated image
        cv2.imwrite(output_path, rotated_image)
    
    print(f"\nCompleted! Rotated images saved to {output_dir}")

if __name__ == "__main__":
    # Process the dataset
    process_dataset(
        source_dir='dataset/labeled',
        output_dir='dataset/rotated',
        min_angle=-10,
        max_angle=10,
        seed=None  # Set to a number for reproducible results
    )
