"""
Script to copy images from 'labeled' folder to 'filtered' folder
based on image names found in 'filtered_rotated' folder.
"""

import os
import shutil
from pathlib import Path


def copy_filtered_images():
    """
    Copy images from labeled folder to filtered folder
    based on filenames in filtered_rotated folder.
    """
    # Define paths
    dataset_dir = Path("dataset")
    filtered_rotated_dir = dataset_dir / "filtered_rotated"
    labeled_dir = dataset_dir / "labeled"
    filtered_dir = dataset_dir / "filtered"
    
    # Create filtered directory if it doesn't exist
    filtered_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all image files from filtered_rotated
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    filtered_rotated_images = [
        f.name for f in filtered_rotated_dir.iterdir()
        if f.is_file() and f.suffix in image_extensions
    ]
    
    print(f"Found {len(filtered_rotated_images)} images in filtered_rotated folder")
    
    # Copy matching images from labeled to filtered
    copied_count = 0
    not_found_count = 0
    
    for image_name in sorted(filtered_rotated_images):
        source_path = labeled_dir / image_name
        dest_path = filtered_dir / image_name
        
        if source_path.exists():
            shutil.copy2(source_path, dest_path)
            copied_count += 1
            print(f"Copied: {image_name}")
        else:
            not_found_count += 1
            print(f"NOT FOUND in labeled folder: {image_name}")
    
    print("\n" + "="*50)
    print(f"Summary:")
    print(f"  Total images in filtered_rotated: {len(filtered_rotated_images)}")
    print(f"  Successfully copied: {copied_count}")
    print(f"  Not found in labeled folder: {not_found_count}")
    print("="*50)


if __name__ == "__main__":
    copy_filtered_images()

