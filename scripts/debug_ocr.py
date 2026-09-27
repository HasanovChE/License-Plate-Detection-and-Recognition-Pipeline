import argparse
from pathlib import Path

import cv2
import pytesseract

from src.detection.detector import LicensePlateDetector
from src.utils.image import crop_with_padding
from src.ocr.preprocessing import preprocess_plate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default="models/best.pt")
    parser.add_argument("--out-dir", default="outputs/debug")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(args.image)

    detector = LicensePlateDetector(model_path=args.model)
    detections = detector.detect(image)

    if not detections:
        print("No detections.")
        return

    for i, det in enumerate(detections):
        bbox = det["bbox"]
        crop = crop_with_padding(image, bbox, padding_ratio=0.15)
        if crop is None:
            continue

        raw_path = out_dir / f"crop_{i}_raw.jpg"
        cv2.imwrite(str(raw_path), crop)

        processed = preprocess_plate(crop)
        proc_path = out_dir / f"crop_{i}_processed.jpg"
        cv2.imwrite(str(proc_path), processed)

        print(f"\n--- Detection {i} | bbox={bbox} | confidence={det['confidence']:.3f} ---")
        print(f"crop size: {crop.shape[1]}x{crop.shape[0]}")
        print(f"Saved raw crop to:       {raw_path}")
        print(f"Saved processed crop to: {proc_path}")

        configs = {
            "whitelist + psm7 (pipeline default)":
                "--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            "no whitelist + psm7": "--oem 3 --psm 7",
            "no whitelist + psm6": "--oem 3 --psm 6",
            "no whitelist + psm11": "--oem 3 --psm 11",
            "raw crop (no preprocessing) + psm7": "--oem 3 --psm 7",
        }

        for label, config in configs.items():
            source = crop if "raw crop" in label else processed
            text = pytesseract.image_to_string(source, config=config)
            print(f"[{label}] -> {text.strip()!r}")


if __name__ == "__main__":
    main()
