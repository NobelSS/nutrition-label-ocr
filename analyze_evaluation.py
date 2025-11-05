"""
Script to analyze evaluation CSV files and identify images with low field_accuracy.
Generates reports per CSV file and aggregated reports per folder.
"""

import os
import csv
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple


def read_csv_file(csv_path: Path) -> List[Dict]:
    """Read a CSV file and return list of records."""
    records = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                row['field_accuracy'] = float(row['field_accuracy'])
                row['value_accuracy'] = float(row['value_accuracy'])
                row['unit_accuracy'] = float(row['unit_accuracy'])
                row['percent_dv_accuracy'] = float(row['percent_dv_accuracy'])
            except (ValueError, KeyError):
                continue
            records.append(row)
    return records


def analyze_csv_file(csv_path: Path, threshold: float = 0.5) -> Dict:
    """Analyze a single CSV file and return statistics."""
    records = read_csv_file(csv_path)
    
    if not records:
        return {
            'total_images': 0,
            'low_accuracy_images': [],
            'low_accuracy_count': 0,
            'average_field_accuracy': 0.0,
            'min_field_accuracy': 0.0,
            'max_field_accuracy': 0.0
        }
    
    low_accuracy_images = [
        {
            'image': r['image'],
            'field_accuracy': r['field_accuracy'],
            'value_accuracy': r['value_accuracy'],
            'unit_accuracy': r['unit_accuracy'],
            'percent_dv_accuracy': r['percent_dv_accuracy']
        }
        for r in records if r['field_accuracy'] < threshold
    ]
    
    field_accuracies = [r['field_accuracy'] for r in records]
    
    return {
        'total_images': len(records),
        'low_accuracy_images': low_accuracy_images,
        'low_accuracy_count': len(low_accuracy_images),
        'average_field_accuracy': sum(field_accuracies) / len(field_accuracies) if field_accuracies else 0.0,
        'min_field_accuracy': min(field_accuracies) if field_accuracies else 0.0,
        'max_field_accuracy': max(field_accuracies) if field_accuracies else 0.0,
        'threshold': threshold
    }


def find_consistent_low_accuracy_images(csv_files: List[Path], threshold: float) -> List[Dict]:
    """Find images that have low accuracy across multiple CSV files."""
    # Track each image across all files
    image_tracker = defaultdict(lambda: {
        'total_appearances': 0,
        'low_accuracy_appearances': 0,
        'details': []  # List of (filename, field_accuracy) tuples
    })
    
    for csv_file in csv_files:
        records = read_csv_file(csv_file)
        for record in records:
            image_name = record['image']
            field_acc = record['field_accuracy']
            
            image_tracker[image_name]['total_appearances'] += 1
            image_tracker[image_name]['details'].append({
                'file': csv_file.name,
                'field_accuracy': field_acc
            })
            
            if field_acc < threshold:
                image_tracker[image_name]['low_accuracy_appearances'] += 1
    
    # Find images that are consistently low (low in all files they appear in)
    consistently_low = []
    for image_name, data in image_tracker.items():
        if data['total_appearances'] > 0:
            consistency_rate = data['low_accuracy_appearances'] / data['total_appearances']
            avg_field_acc = sum(d['field_accuracy'] for d in data['details']) / len(data['details'])
            
            consistently_low.append({
                'image': image_name,
                'total_files': data['total_appearances'],
                'low_accuracy_files': data['low_accuracy_appearances'],
                'consistency_rate': consistency_rate,
                'average_field_accuracy': avg_field_acc,
                'min_field_accuracy': min(d['field_accuracy'] for d in data['details']),
                'max_field_accuracy': max(d['field_accuracy'] for d in data['details']),
                'details': data['details']
            })
    
    # Sort by consistency rate (descending), then by average field accuracy (ascending)
    consistently_low.sort(key=lambda x: (-x['consistency_rate'], x['average_field_accuracy']))
    
    return consistently_low


def analyze_folder(folder_path: Path, threshold: float = 0.3) -> Dict:
    """Analyze all CSV files in a folder."""
    csv_files = sorted(folder_path.glob('*.csv'))
    
    if not csv_files:
        return {
            'folder_name': folder_path.name,
            'csv_files': [],
            'file_analyses': {},
            'aggregated_stats': {},
            'consistent_low_accuracy': []
        }
    
    file_analyses = {}
    all_field_accuracies = []
    all_low_accuracy_images = []
    total_images = 0
    
    for csv_file in csv_files:
        analysis = analyze_csv_file(csv_file, threshold)
        file_analyses[csv_file.name] = analysis
        
        # Collect data for aggregated stats
        records = read_csv_file(csv_file)
        for record in records:
            all_field_accuracies.append(record['field_accuracy'])
            total_images += 1
            if record['field_accuracy'] < threshold:
                all_low_accuracy_images.append({
                    'file': csv_file.name,
                    'image': record['image'],
                    'field_accuracy': record['field_accuracy']
                })
    
    # Find consistently low accuracy images
    consistent_low_accuracy = find_consistent_low_accuracy_images(csv_files, threshold)
    
    # Calculate aggregated statistics
    aggregated_stats = {
        'total_csv_files': len(csv_files),
        'total_images': total_images,
        'total_low_accuracy_images': len(all_low_accuracy_images),
        'percentage_low_accuracy': (len(all_low_accuracy_images) / total_images * 100) if total_images > 0 else 0.0,
        'average_field_accuracy': sum(all_field_accuracies) / len(all_field_accuracies) if all_field_accuracies else 0.0,
        'min_field_accuracy': min(all_field_accuracies) if all_field_accuracies else 0.0,
        'max_field_accuracy': max(all_field_accuracies) if all_field_accuracies else 0.0,
        'threshold': threshold
    }
    
    return {
        'folder_name': folder_path.name,
        'csv_files': [f.name for f in csv_files],
        'file_analyses': file_analyses,
        'aggregated_stats': aggregated_stats,
        'all_low_accuracy_images': sorted(all_low_accuracy_images, key=lambda x: x['field_accuracy']),
        'consistent_low_accuracy': consistent_low_accuracy
    }


def generate_text_report(analysis: Dict, output_path: Path):
    """Generate a human-readable text report."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"Evaluation Report for Folder: {analysis['folder_name']}\n")
        f.write("=" * 80 + "\n\n")
        
        # Aggregated statistics
        agg = analysis['aggregated_stats']
        f.write("AGGREGATED STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Threshold: {agg['threshold']}\n")
        f.write(f"Total CSV Files: {agg['total_csv_files']}\n")
        f.write(f"Total Images: {agg['total_images']}\n")
        f.write(f"Images with Low Field Accuracy (< {agg['threshold']}): {agg['total_low_accuracy_images']}\n")
        f.write(f"Percentage with Low Accuracy: {agg['percentage_low_accuracy']:.2f}%\n")
        f.write(f"Average Field Accuracy: {agg['average_field_accuracy']:.4f}\n")
        f.write(f"Min Field Accuracy: {agg['min_field_accuracy']:.4f}\n")
        f.write(f"Max Field Accuracy: {agg['max_field_accuracy']:.4f}\n\n")
        
        # Consistently low accuracy images
        if analysis['consistent_low_accuracy']:
            f.write("CONSISTENTLY LOW ACCURACY IMAGES\n")
            f.write("-" * 80 + "\n")
            f.write("Images that appear in multiple CSV files with low field_accuracy\n")
            f.write("Sorted by consistency rate (how often they're low) and average accuracy\n\n")
            
            # Filter to show only images that appear in multiple files
            multi_file_images = [img for img in analysis['consistent_low_accuracy'] 
                               if img['total_files'] > 1]
            
            if multi_file_images:
                # Show only images that have average field_accuracy below threshold
                filtered_images = [img for img in multi_file_images 
                                 if img['average_field_accuracy'] < agg['threshold']]
                
                f.write(f"Images appearing in multiple files with low accuracy: {len(filtered_images)}\n\n")
                
                for img in filtered_images:
                    consistency_pct = img['consistency_rate'] * 100
                    f.write(f"  {img['image']:30s} | "
                           f"Files: {img['low_accuracy_files']}/{img['total_files']} "
                           f"({consistency_pct:.1f}%) | "
                           f"Avg: {img['average_field_accuracy']:.4f} | "
                           f"Min: {img['min_field_accuracy']:.4f} | "
                           f"Max: {img['max_field_accuracy']:.4f}\n")
                    # Show details for each file
                    for detail in sorted(img['details'], key=lambda x: x['field_accuracy']):
                        status = "LOW" if detail['field_accuracy'] < agg['threshold'] else "OK"
                        f.write(f"    - {detail['file']:50s} | "
                               f"{detail['field_accuracy']:.4f} ({status})\n")
                    f.write("\n")
            else:
                f.write("No images appear in multiple CSV files.\n")
            f.write("\n")
        
        # All low accuracy images across all files
        if analysis['all_low_accuracy_images']:
            f.write("ALL LOW ACCURACY IMAGES (Sorted by Field Accuracy)\n")
            f.write("-" * 80 + "\n")
            for img in analysis['all_low_accuracy_images']:
                f.write(f"  {img['file']:50s} | {img['image']:30s} | {img['field_accuracy']:.4f}\n")
            f.write("\n")
        
        # Per-file analysis
        f.write("PER-FILE ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        
        for csv_filename, file_analysis in sorted(analysis['file_analyses'].items()):
            f.write(f"File: {csv_filename}\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Total Images: {file_analysis['total_images']}\n")
            f.write(f"  Low Accuracy Images: {file_analysis['low_accuracy_count']}\n")
            f.write(f"  Average Field Accuracy: {file_analysis['average_field_accuracy']:.4f}\n")
            f.write(f"  Min Field Accuracy: {file_analysis['min_field_accuracy']:.4f}\n")
            f.write(f"  Max Field Accuracy: {file_analysis['max_field_accuracy']:.4f}\n")
            
            if file_analysis['low_accuracy_images']:
                f.write(f"\n  Low Accuracy Images (< {file_analysis['threshold']}):\n")
                # Sort by field_accuracy
                sorted_images = sorted(file_analysis['low_accuracy_images'], 
                                     key=lambda x: x['field_accuracy'])
                for img in sorted_images:
                    f.write(f"    - {img['image']:30s} | Field: {img['field_accuracy']:.4f} | "
                           f"Value: {img['value_accuracy']:.4f} | "
                           f"Unit: {img['unit_accuracy']:.4f} | "
                           f"Percent DV: {img['percent_dv_accuracy']:.4f}\n")
            else:
                f.write(f"\n  No low accuracy images found.\n")
            
            f.write("\n")


def generate_json_report(analysis: Dict, output_path: Path):
    """Generate a JSON report for programmatic access."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)


def main():
    """Main function to analyze evaluation folders."""
    parser = argparse.ArgumentParser(
        description='Analyze evaluation CSV files and identify images with low field_accuracy'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.3,
        help='Threshold for field_accuracy (default: 0.3). Images below this threshold are considered low accuracy.'
    )
    parser.add_argument(
        '--folder',
        type=str,
        default=None,
        help='Specific folder to analyze (e.g., svtrv2_mobile). If not specified, all folders will be analyzed.'
    )
    
    args = parser.parse_args()
    threshold = args.threshold
    evaluation_dir = Path('evaluation')
    
    if not evaluation_dir.exists():
        print(f"Error: {evaluation_dir} directory not found!")
        return
    
    # Find all subdirectories in evaluation
    all_subdirs = [d for d in evaluation_dir.iterdir() if d.is_dir()]
    
    # Filter by specific folder if provided
    if args.folder:
        subdirs = [d for d in all_subdirs if d.name == args.folder]
        if not subdirs:
            print(f"Error: Folder '{args.folder}' not found in evaluation directory!")
            print(f"Available folders: {[d.name for d in all_subdirs]}")
            return
    else:
        subdirs = all_subdirs
    
    if not subdirs:
        print("No subdirectories found in evaluation folder!")
        return
    
    print(f"Found {len(subdirs)} folder(s) to analyze...")
    print(f"Using threshold: {threshold}\n")
    
    # Process each folder separately
    for folder in sorted(subdirs):
        print(f"Processing folder: {folder.name}")
        
        # Analyze the folder
        analysis = analyze_folder(folder, threshold)
        
        if not analysis['csv_files']:
            print(f"  No CSV files found in {folder.name}, skipping...\n")
            continue
        
        # Create output directory for reports
        output_dir = Path('evaluation_reports') / folder.name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate reports
        text_report_path = output_dir / 'low_accuracy_report.txt'
        json_report_path = output_dir / 'low_accuracy_report.json'
        
        generate_text_report(analysis, text_report_path)
        generate_json_report(analysis, json_report_path)
        
        print(f"  Generated reports:")
        print(f"    - {text_report_path}")
        print(f"    - {json_report_path}")
        print(f"  Total images: {analysis['aggregated_stats']['total_images']}")
        print(f"  Low accuracy images: {analysis['aggregated_stats']['total_low_accuracy_images']}")
        print()
    
    print("Analysis complete!")


if __name__ == '__main__':
    main()

