import re
from constants import FIELD_REGEX_PATTERNS, FIELDS, ALL_UNITS_PATTERN

def clean_ocr_units(text):
    """
    Fix common OCR errors that appear **before units** in nutrition labels.
    - O, o → 0
    - l → 1
    - S → 5
    Only when immediately followed by a unit like g, mg, kkal, ml, etc.
    """
    
    text = re.sub(rf'\b[Oo](?=({ALL_UNITS_PATTERN}))', '0', text)
    text = re.sub(rf'\bl(?=({ALL_UNITS_PATTERN}))', '1', text)
    text = re.sub(rf'\bS(?=({ALL_UNITS_PATTERN}))', '5', text)
    
    # Remove extra symbols and normalize spaces
    text = re.sub(r'\([^)]*\)', ' ', text)
    text = re.sub(r'[^\w\s\.\-%]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def parse_nutrition_text(text):
    result = {}
    cleaned = clean_ocr_units(text)
    text_lower = cleaned.lower()
    
    print(f"Cleaned Text for Parsing:\n{cleaned}\n")
    
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
