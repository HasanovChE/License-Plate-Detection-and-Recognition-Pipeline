import cv2
import os
import time
from src.pipeline.pipeline import LicensePlatePipeline
from src.database.parking_log import ParkingLog

os.makedirs("outputs/videos", exist_ok=True)

pipeline = LicensePlatePipeline(
    model_path="models/best.pt"
)

db = ParkingLog(db_path="parking.db")

video_path = "dashcam.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {video_path}")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("outputs/videos/result.mp4", fourcc, fps, (width, height))

frame_skip = 3
frame_count = 0
last_result = None

last_seen = {}

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    current_time = time.time()

    if frame_count % frame_skip == 0:
        last_result = pipeline.process(frame)

    if last_result and "plates" in last_result:
        for plate_data in last_result["plates"]:
            x1, y1, x2, y2 = plate_data["bbox"]
            text = plate_data["plate"]

            if text not in last_seen or (current_time - last_seen[text] > 30):
                country = plate_data.get("format", {}).get("country", "UNKNOWN")
                confidence = plate_data.get("detection_confidence", 0.0)
                
                db.insert(
                    plate=text,
                    country=country,
                    confidence=confidence
                )
                last_seen[text] = current_time

            cv2.rectangle(
                frame, 
                (x1, y1), 
                (x2, y2), 
                (0, 255, 0), 
                2
            )
            
            cv2.putText(
                frame, 
                text, 
                (x1, max(20, y1 - 10)), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 255, 0), 
                2
            )

    out.write(frame)
    cv2.imshow("Video Processing", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_count += 1

cap.release()
out.release()
cv2.destroyAllWindows()