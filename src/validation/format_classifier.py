import re

FORMATS = {
    "AZERBAIJAN": re.compile(
        r"^\d{2}[A-Z]{2}\d{3}$"
    ),
    "UK": re.compile(
        r"^[A-Z]{2}\d{2}[A-Z]{3}$"
    ),
    "GERMANY": re.compile(
        r"^[A-Z]{1,3}[A-Z]{1,2}\d{1,4}$"
    ),
    "POLAND": re.compile(
        r"^[A-Z]{2,3}\d{4,5}$"
    )
}

def classify_plate(text: str):
    text = re.sub(
        r"[^A-Z0-9]",
        "",
        text.upper()
    )
    matches = []
    
    for country, pattern in FORMATS.items():
        if pattern.fullmatch(text):
            matches.append(country)
            
    if not matches:
        return {
            "country": "UNKNOWN",
            "confidence": 0.0
        }
        
    if len(matches) == 1:
        return {
            "country": matches[0],
            "confidence": 1.0
        }
        
    return {
        "country": matches[0],
        "confidence": 0.5
    }