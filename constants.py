import re
from enum import Enum

class PipelineVariation(Enum):
    """Enum for different pipeline variations"""
    FULL_PIPELINE = "full_pipeline"
    NO_PIPELINE = "no_pipeline"
    NO_OBJECT_DETECTION = "no_object_detection"
    NO_RECTIFICATION = "no_rectification"
    NO_DESKEW = "no_deskew"
    NO_DESKEW_NO_RECTIFICATION = "no_deskew_no_rectification"
    NO_PREPROCESS = "no_preprocess"

FIELDS = [
    # --- Serving Information ---
    ("serving_size", ["takaran saji", "serving size"], ["g", "ml"]),
    ("number_of_servings", ["sajian per kemasan", "servings per container"], []),

    # --- Energy ---
    ("energy_total", ["energi total", "total energy", "kalori"], ["kkal", "kcal"]),
    ("energy_from_fat", ["energi dari lemak", "calories from fat"], ["kkal", "kcal"]),
    ("energy_from_fat_saturated", ["energi dari lemak jenuh", "calories from saturated fat"], ["kkal", "kcal"]),

    # --- Fat ---
    ("fat_total", ["lemak total", "total fat"], ["g"]),
    ("fat_saturated", ["lemak jenuh", "saturated fat"], ["g"]),
    ("fat_trans", ["lemak trans", "trans fat"], ["g"]),
    ("cholesterol", ["kolestrol", "cholesterol"], ["mg"]),

    # --- Protein & Carbs ---
    ("protein", ["protein"], ["g"]),
    ("carbohydrate_total", ["karbohidrat total", "total carbohydrate", "karbohidrat"], ["g"]),
    ("sugar", ["gula", "sugar", "sugars"], ["g"]),
    ("sucrose", ["sukrosa"], ["g"]),
    ("lactose", ["laktosa"], ["g"]),
    ("dietary_fiber", ["serat pangan", "dietary fiber", "fiber"], ["g"]),

    # --- Minerals ---
    ("sodium", ["garam", "natrium", "salt", "sodium"], ["mg"]),
    ("potassium", ["kalium"], ["mg"]),
    ("calcium", ["kalsium"], ["mg"]),
    ("iron", ["zat besi", "besi", "iron"], ["mg"]),
    ("zinc", ["seng", "zinc"], ["mg"]),
    ("magnesium", ["magnesium"], ["mg"]),
    ("phosphorus", ["fosfor", "phosphorus", "phosphor"], ["mg"]),
    ("iodine", ["yodium", "iodium", "iodine"], ["mcg", "µg"]),
    ("selenium", ["selenium"], ["mcg", "µg"]),
    ("chromium", ["kromium", "chromium"], ["mcg", "µg"]),

    # --- Vitamins ---
    ("vitamin_a", ["vitamin a"], ["mcg", "µg", "IU"]),
    ("vitamin_b1", ["vitamin b1", "tiamin", "thiamine"], ["mg"]),
    ("vitamin_b2", ["vitamin b2", "riboflavin"], ["mg"]),
    ("vitamin_b3", ["vitamin b3", "niasin", "niacin"], ["mcg", "mg"]),
    ("vitamin_b5", ["vitamin b5", "asam pantotenat", "pantothenic acid"], ["mg"]),
    ("vitamin_b6", ["vitamin b6", "piridoksin", "pyridoxine"], ["mg"]),
    ("vitamin_b7", ["vitamin b7", "biotin"], ["mcg"]),
    ("vitamin_b12", ["vitamin b12", "kobalamin", "cobalamin"], ["mcg", "µg"]),
    ("vitamin_c", ["vitamin c", "asam askorbat", "ascorbic acid"], ["mg"]),
    ("vitamin_d", ["vitamin d"], ["mcg", "µg", "IU"]),
    ("vitamin_d3", ["vitamin d3"], ["mcg", "µg", "IU"]),
    ("vitamin_e", ["vitamin e"], ["mg", "IU"]),
    ("folic_acid", ["asam folat", "folic acid", "vitamin b9"], ["mcg", "µg"]),
]


NUMBER_BEFORE_LABEL_FIELDS = {"number_of_servings"}

FIELD_REGEX_PATTERNS = {}
for canonical, synonyms, units in FIELDS:
    synonyms_pattern = "|".join([re.escape(s) for s in synonyms])
    units_pattern = "|".join([re.escape(u) for u in units]) if units else r"\w*"

    if canonical in NUMBER_BEFORE_LABEL_FIELDS:
        # Pattern: number comes BEFORE label
        if units:
            units_pattern = "|".join([re.escape(u) for u in units])
            regex = rf"([\d\.]+)\s*(?:{units_pattern})?\s*(?:{synonyms_pattern})"
        else:
            regex = rf"([\d\.]+)\s*(?:{synonyms_pattern})"
    else:
        # Pattern: label comes BEFORE number
        if units:
            units_pattern = "|".join([re.escape(u) for u in units])
            regex = rf"(?:{synonyms_pattern})\s*.{{0,10}}?([\d\.]+)\s*(?:{units_pattern})?"
        else:
            regex = rf"(?:{synonyms_pattern})\s*.{{0,10}}?([\d\.]+)"
    
    # if canonical in NUMBER_BEFORE_LABEL_FIELDS:
    #     # Number usually before the label
    #     regex = rf"([\d\.]+)\s*(?:{units_pattern})?\s*(?:{synonyms_pattern})"
    # else:
    #     # Label comes before number
    #     regex = rf"(?:{synonyms_pattern})\s*.{{0,10}}?([\d\.]+)\s*(?:{units_pattern})?"
    
    FIELD_REGEX_PATTERNS[canonical] = regex
    
    
ALL_UNITS = set()
for _, _, units in FIELDS:
    ALL_UNITS.update(units)

# Create the regex pattern for units
ALL_UNITS_PATTERN = "|".join([re.escape(u) for u in ALL_UNITS if u])