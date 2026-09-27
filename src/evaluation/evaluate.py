import os
import cv2
import pandas as pd
from pathlib import Path
from src.detection.detector import LicensePlateDetector
from src.ocr.tesseract_ocr import TesseractOCR
from src.evaluation.detection_metrics import calculate_iou
from src.evaluation.ocr_metrics import exact_match, character_error_rate

def main():
    model_path = "models/best.pt"
    detector = LicensePlateDetector(model_path=model_path)
    ocr = TesseractOCR()
    
    test_images_dir = Path("data/images/test")
    
    if not test_images_dir.exists():
        print(f"Test directory not found: {test_images_dir}")
        return

    os.makedirs("outputs/evaluation", exist_ok=True)
    
    rows = []
    ious = []
    
    for img_path in test_images_dir.glob("*.jpg"):
        image = cv2.imread(str(img_path))
        if image is None:
            continue
            
        detections = detector.detect(image)
        
        predicted_box = detections[0]["bbox"] if detections else [0, 0, 0, 0]
        predicted_text = "UNKNOWN"
        
        if detections:
            x1, y1, x2, y2 = predicted_box
            crop = image[y1:y2, x1:x2]
            if crop.size > 0:
                predicted_text = ocr.read(crop).strip()
        
        ground_truth_box = [0, 0, 0, 0] 
        ground_truth_plate = "99AB123"   
        
        iou = calculate_iou(predicted_box, ground_truth_box)
        ious.append(iou)
        
        is_correct = exact_match(predicted_text, ground_truth_plate)
        cer = character_error_rate(predicted_text, ground_truth_plate)
        
        rows.append({
            "image": img_path.name,
            "detected_box": str(predicted_box),
            "ocr_text": predicted_text,
            "ground_truth": ground_truth_plate,
            "correct": is_correct,
            "iou": iou,
            "cer": cer
        })
        
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv("outputs/evaluation/results.csv", index=False)
        
        mean_iou = sum(ious) / len(ious) if ious else 0.0
        accuracy = df["correct"].mean()
        
        print("========================================")
        print("LICENSE PLATE PIPELINE EVALUATION")
        print("========================================")
        print(f"Mean IoU: {mean_iou:.2f}")
        print(f"End-to-end accuracy: {accuracy:.2%}")
        print("Results saved to outputs/evaluation/results.csv")
    else:
        print("No evaluation results generated. Check test images.")

if __name__ == "__main__":
    main()