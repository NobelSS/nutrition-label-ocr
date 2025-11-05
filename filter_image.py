import json
import os
import shutil

def get_low_accuracy_images_from_full_pipeline(json_file_path):
    """
    Loads a JSON report and extracts unique low-accuracy image names
    ONLY from the 'evaluation_report_full_pipeline.csv' analysis.
    """
    if not os.path.exists(json_file_path):
        print(f"Error: JSON file not found at {json_file_path}")
        return None
    
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        image_set = set()
        
        file_analyses = data.get("file_analyses", {})
        pipeline_analysis = file_analyses.get("evaluation_report_full_pipeline.csv")
        
        if not pipeline_analysis:
            print(f"Warning: 'evaluation_report_full_pipeline.csv' not found in file_analyses of {json_file_path}")
            return image_set

        low_accuracy_images = pipeline_analysis.get("low_accuracy_images", [])
        for img_data in low_accuracy_images:
            if "image" in img_data:
                image_set.add(img_data["image"])
                    
        return image_set
        
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_file_path}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while processing {json_file_path}: {e}")
        return None

def copy_filtered_images(exclusion_list, source_dir, target_dir):
    """
    Copies all files from source_dir to target_dir, EXCEPT for those
    in the exclusion_list.
    """
    
    if not os.path.isdir(source_dir):
        print(f"Error: Source directory '{source_dir}' not found. No files will be copied.")
        return

    if not os.path.exists(target_dir):
        try:
            os.makedirs(target_dir)
            print(f"Created target directory: {target_dir}")
        except OSError as e:
            print(f"Error creating target directory {target_dir}: {e}")
            return
            
    copied_count = 0
    skipped_count = 0
    
    try:
        all_source_files = os.listdir(source_dir)
        print(f"Found {len(all_source_files)} total files in {source_dir}.")
        
        for image_name in all_source_files:
            source_path = os.path.join(source_dir, image_name)
            
            if not os.path.isfile(source_path):
                continue
            
            if image_name in exclusion_list:
                skipped_count += 1
            else:
                target_path = os.path.join(target_dir, image_name)
                try:
                    shutil.copy2(source_path, target_path)
                    copied_count += 1
                except Exception as e:
                    print(f"Error copying {image_name}: {e}")
                    
        print(f"\n--- Filter & Copy Operation Summary ---")
        print(f"Successfully copied: {copied_count} files")
        print(f"Skipped (common):   {skipped_count} files")
        print(f"Total processed:    {copied_count + skipped_count} files")

    except Exception as e:
        print(f"An error occurred while reading source directory {source_dir}: {e}")

if __name__ == "__main__":
    svtr_json_path = "evaluation_reports/svtrv2_mobile/low_accuracy_report.json"
    tesseract_json_path = "evaluation_reports/tesseract/low_accuracy_report.json"
    source_image_dir = "dataset/rotated"
    target_image_dir = "dataset/filtered"

    svtr_images = get_low_accuracy_images_from_full_pipeline(svtr_json_path)
    tesseract_images = get_low_accuracy_images_from_full_pipeline(tesseract_json_path)

    if svtr_images is not None and tesseract_images is not None:
        print(f"Found {len(svtr_images)} unique low-accuracy images from 'full_pipeline' for svtrv2_mobile.")
        print(f"Found {len(tesseract_images)} unique low-accuracy images from 'full_pipeline' for tesseract.")
        
        common_low_acc_images = svtr_images.intersection(tesseract_images)
        
        if common_low_acc_images:
            print(f"\nFound {len(common_low_acc_images)} common low-accuracy images to exclude:")
            print("---------------------------------")
            for img in enumerate(sorted(list(common_low_acc_images))):
                print(img)
            print("---------------------------------")
            
        else:
            print("\nNo common low-accuracy images were found. All files will be copied.")
        
        copy_filtered_images(common_low_acc_images, source_image_dir, target_image_dir)
        
    else:
        print("\nCould not proceed with filtering. Please ensure both JSON files are present at:")
        print(f"1. {svtr_json_path}")
        print(f"2. {tesseract_json_path}")
        print("And that both contain the 'evaluation_report_full_pipeline.csv' analysis.")
