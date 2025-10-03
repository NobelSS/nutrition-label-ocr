import re

FIELDS = [
    ("serving_size", ["takaran saji", "serving size"], ["g", "ml"]),
    ("number_of_servings", ["sajian per kemasan", "servings per container"], []),
    ("energy_total", ["energi total", "total energy", "kalori"], ["kkal", "kcal"]),
    ("energy_from_fat", ["energi dari lemak", "calories from fat"], ["kkal", "kcal"]),
    ("energy_from_fat_saturated", ["energi dari lemak jenuh", "calories from saturated fat"], ["kkal", "kcal"]),
    ("fat_total", ["lemak total", "total fat"], ["g"]),
    ("fat_saturated", ["lemak jenuh", "saturated fat"], ["g"]),
    ("fat_trans", ["lemak trans", "trans fat"], ["g"]),
    ("protein", ["protein"], ["g"]),
    ("carbohydrate_total", ["karbohidrat total", "total carbohydrate", "karbohidrat"], ["g"]),
    ("sugar", ["gula", "sugar", "sugars"], ["g"]),
    ("dietary_fiber", ["serat pangan", "dietary fiber", "fiber"], ["g"]),
    ("sodium", ["garam", "natrium", "salt", "sodium"], ["mg"]),
    ("iron", ["zat besi"], ["mg"]),
    ("zinc", ["seng"], ["mg"]),
    ("sucrose", ["sukrosa"], ["g"]),
    ("lactose", ["laktosa"], ["g"]),
    ("potassium", ["kalium"], ["mg"]),
    ("calcium", ["kalsium"], ["mg"]),
    ("cholesterol", ["kolestrol", "cholesterol"], ["mg"])
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