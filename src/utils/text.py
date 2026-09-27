import re

def clean_ocr_text(text: str) -> str:
    text = text.upper()
    text = re.sub(
        r"[^A-Z0-9]",
        "",
        text
    )
    
    replacements = {
        "O": "0",
        "I": "1"
    }
    
    cleaned = []
    for char in text:
        cleaned.append(
            replacements.get(char, char)
        )
        
    return "".join(cleaned)