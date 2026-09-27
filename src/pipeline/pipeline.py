import logging

import cv2

from src.detection.detector import LicensePlateDetector
from src.ocr.preprocessing import preprocess_plate
from src.ocr.tesseract_ocr import TesseractOCR
from src.utils.image import crop_with_padding
from src.utils.text import clean_ocr_text
from src.validation.format_classifier import classify_plate

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s "
        "%(levelname)s "
        "%(message)s"
    )
)

logger = logging.getLogger(__name__)


class LicensePlatePipeline:
    def __init__(
        self,
        model_path,
        tesseract_path=None,
        confidence=0.5
    ):
        self.detector = LicensePlateDetector(
            model_path=model_path,
            confidence=confidence
        )
        self.ocr = TesseractOCR(
            tesseract_path=tesseract_path
        )

    def process(self, image):
        detections = self.detector.detect(image)
        results = []

        if not detections:
            return {
                "success": False,
                "plates": [],
                "message": "No license plate detected"
            }

        for detection in detections:
            bbox = detection["bbox"]
            crop = crop_with_padding(
                image,
                bbox,
                padding_ratio=0.15
            )

            if crop is None:
                continue

            processed = preprocess_plate(crop)
            raw_text = self.ocr.read(
                processed
            )
            cleaned_text = clean_ocr_text(
                raw_text
            )
            format_result = classify_plate(
                cleaned_text
            )

            logger.info(
                "Detected plate: %s",
                cleaned_text
            )

            results.append({
                "bbox": bbox,
                "detection_confidence":
                    detection["confidence"],
                "raw_ocr": raw_text.strip(),
                "plate": cleaned_text,
                "format": format_result,
                "crop": crop
            })

        return {
            "success": len(results) > 0,
            "plates": results
        }
