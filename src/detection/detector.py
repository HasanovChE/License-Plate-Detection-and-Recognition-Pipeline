from pathlib import Path
from typing import List, Dict
from ultralytics import YOLO

class LicensePlateDetector:
    def __init__(
        self,
        model_path: str,
        confidence: float = 0.35,
        iou_threshold: float = 0.45
    ):
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}"
            )
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.iou_threshold = iou_threshold

    def detect(self, image):
        results = self.model.predict(
            source=image,
            conf=self.confidence,
            iou=self.iou_threshold,
            verbose=False
        )
        detections = []
        if not results:
            return detections
        
        result = results[0]
        if result.boxes is None:
            return detections
            
        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            confidence = float(box.conf[0].cpu().item())
            class_id = int(box.cls[0].cpu().item())
            detections.append({
                "bbox": [int(x) for x in xyxy],
                "confidence": confidence,
                "class_id": class_id
            })
            
        return detections