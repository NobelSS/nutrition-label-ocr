import re
import csv
from constants import FIELD_REGEX_PATTERNS, FIELDS, ALL_UNITS_PATTERN

def clean_ocr_units(text):
    """
    Fix common OCR errors that appear **before units** in nutrition labels.
    - O, o → 0
    - l → 1
    - S → 5
    Only when immediately followed by a unit like g, mg, kkal, ml, etc.
    """
    if not isinstance(text, str):
        if hasattr(text, "text"):
            text = text.text
        elif isinstance(text, list):
            text = " ".join(
                t[1][0] if isinstance(t, list) and len(t) > 1 else str(t)
                for t in text
            )
        else:
            text = str(text)

    text = re.sub(rf'\b[Oo](?=({ALL_UNITS_PATTERN}))', '0', text)
    text = re.sub(rf'\bl(?=({ALL_UNITS_PATTERN}))', '1', text)
    text = re.sub(rf'\bS(?=({ALL_UNITS_PATTERN}))', '5', text)

    text = re.sub(r'\([^)]*\)', ' ', text)
    text = re.sub(r'[^\w\s\.\-%]', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def parse_nutrition_text(text):
    result = {}
    cleaned = clean_ocr_units(text)
    text_lower = cleaned.lower()
    
    # print(f"Cleaned Text for Parsing:\n{cleaned}\n")
    
    for field, (canonical, synonyms, units) in zip(FIELD_REGEX_PATTERNS.keys(), FIELDS):
        pattern = FIELD_REGEX_PATTERNS[field]
        match = re.search(pattern, text_lower, re.IGNORECASE)
        
        if match:
            try:
                value_str = match.group(1)
                unit_str = match.group(2) if match.lastindex and match.lastindex >= 2 else None
                
                if value_str:
                    # Convert to int if no decimal point, else float
                    if "." in value_str:
                        value = float(value_str)
                    else:
                        value = int(value_str)
                    data = {"value": value}
                
                # Use OCR unit if present, otherwise default unit
                if unit_str and unit_str.strip():
                    data["unit"] = unit_str.strip()
                elif units:
                    data["unit"] = units[0]
                
                # Look for percentage (% or percent_dv) after the value
                # Pattern: look for percentage within next 20 characters after the match
                match_end = match.end()
                text_after = text_lower[match_end:match_end + 10]
                percent_match = re.search(r'^\s*(\d+)\s*%', text_after)
                
                if percent_match:
                    data["percent_dv"] = int(percent_match.group(1))
                
                result[field] = data
            except (ValueError, IndexError):
                continue
    
    return result

def evaluate(parsed, ground_truth):
    """
    Compare parsed OCR nutrition data with ground truth.

    Args:
        parsed (dict): OCR parsed output.
        ground_truth (dict): True label data.

    Returns:
        dict: Comparison report.
    """
    report = {
        "matches": {},
        "differences": {},
        "missing_in_parsed": [],
        "missing_in_ground_truth": []
    }
    
    all_fields = set(parsed.keys()) | set(ground_truth.keys())
    
    for field in all_fields:
        parsed_val = parsed.get(field)
        truth_val = ground_truth.get(field)
        
        if parsed_val and truth_val:
            # Compare value
            value_match = parsed_val.get("value") == truth_val.get("value")
            # Compare unit if present
            unit_match = parsed_val.get("unit") == truth_val.get("unit")
            # Compare percent_dv if present
            percent_match = parsed_val.get("percent_dv") == truth_val.get("percent_dv")
            
            if value_match and unit_match and percent_match:
                report["matches"][field] = parsed_val
            else:
                report["differences"][field] = {
                    "parsed": parsed_val,
                    "ground_truth": truth_val
                }
        elif parsed_val and not truth_val:
            report["missing_in_ground_truth"].append(field)
        elif truth_val and not parsed_val:
            report["missing_in_parsed"].append(field)
    
    return report

def evaluate_metric(parsed, ground_truth):
    """
    Returns numeric evaluation metrics comparing parsed OCR output vs ground truth.
    """
    total_fields = len(ground_truth)
    correct_fields = 0
    value_correct = 0
    unit_correct = 0
    percent_correct = 0
    
    total_values = 0
    total_units = 0
    total_percent = 0
    
    for field, truth_val in ground_truth.items():
        parsed_val = parsed.get(field)
        if not parsed_val:
            continue  # missing field → no credit
        
        # --- value ---
        if "value" in truth_val:
            total_values += 1
            if parsed_val.get("value") == truth_val.get("value"):
                value_correct += 1
        
        # --- unit ---
        if "unit" in truth_val:
            total_units += 1
            if parsed_val.get("unit") == truth_val.get("unit"):
                unit_correct += 1
        
        # --- percent_dv ---
        if "percent_dv" in truth_val:
            total_percent += 1
            if parsed_val.get("percent_dv") == truth_val.get("percent_dv"):
                percent_correct += 1
        
        # --- strict match for field accuracy ---
        value_match = parsed_val.get("value") == truth_val.get("value")
        unit_match = (
            parsed_val.get("unit") == truth_val.get("unit")
            if "unit" in truth_val else True
        )
        percent_match = (
            parsed_val.get("percent_dv") == truth_val.get("percent_dv")
            if "percent_dv" in truth_val else True
        )
        
        if value_match and unit_match and percent_match:
            correct_fields += 1
            
    return {
        "field_accuracy": round(correct_fields / total_fields, 2) if total_fields else 0,
        "value_accuracy": round(value_correct / total_values, 2) if total_values else 0,
        "unit_accuracy": round(unit_correct / total_units, 2) if total_units else 0,
        "percent_dv_accuracy": round(percent_correct / total_percent, 2) if total_percent else 0
    }

def export_to_csv(data, filepath):    
    fieldnames = ["image", "field_accuracy", "value_accuracy", "unit_accuracy", "percent_dv_accuracy"]
    
    with open(filepath, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for image, metrics in data.items():
            row = {
                "image": image,
                "field_accuracy": round(metrics.get("field_accuracy", 0), 2),
                "value_accuracy": round(metrics.get("value_accuracy", 0), 2),
                "unit_accuracy": round(metrics.get("unit_accuracy", 0), 2),
                "percent_dv_accuracy": round(metrics.get("percent_dv_accuracy", 0), 2),
            }
            writer.writerow(row)