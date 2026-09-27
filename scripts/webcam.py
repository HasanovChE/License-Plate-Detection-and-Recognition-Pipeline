import cv2
import time
from src.pipeline.pipeline import LicensePlatePipeline
from src.database.parking_log import ParkingLog

pipeline = LicensePlatePipeline( 
    model_path="models/best.pt" 
) 

db = ParkingLog(db_path="parking.db")

cap = cv2.VideoCapture(0) 
 
if not cap.isOpened(): 
    raise RuntimeError( 
        "Cannot open webcam" 
    ) 

last_seen = {}
 
while True: 
    ret, frame = cap.read() 
 
    if not ret: 
        break 
 
    result = pipeline.process(frame) 
    current_time = time.time()
 
    if result.get("success") and "plates" in result:
        for plate_data in result["plates"]: 
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
 
    cv2.imshow( 
        "License Plate Recognition", 
        frame 
    ) 
 
    if cv2.waitKey(1) & 0xFF == ord("q"): 
        break 

cap.release() 
cv2.destroyAllWindows()