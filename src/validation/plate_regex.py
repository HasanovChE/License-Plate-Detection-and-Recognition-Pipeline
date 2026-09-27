import re

AZ_PLATE_PATTERN = re.compile(
    r"^\d{2}[A-Z]{2}\d{3}$"
)

def normalize_az_plate(text: str) -> str:
    text = text.upper()
    text = text.replace("-", "")
    text = text.replace(" ", "")
    text = text.replace(".", "")
    return text

def validate_az_plate(text: str) -> bool:
    text = normalize_az_plate(text)
    return bool(
        AZ_PLATE_PATTERN.fullmatch(text)
    )