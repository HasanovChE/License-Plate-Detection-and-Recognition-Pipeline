from fastapi import FastAPI, UploadFile, File 
import cv2 
import numpy as np 
 
from src.pipeline.pipeline import LicensePlatePipeline 
 
app = FastAPI( 
    title="License Plate Recognition API", 
    version="1.0.0" 
) 
 
pipeline = LicensePlatePipeline( 
    model_path="models/best.pt" ,
     confidence=0.5
) 
 
@app.get("/health") 
def health(): 
 
    return { 
        "status": "ok" 
    } 
 
@app.post("/predict") 
async def predict( 
    file: UploadFile = File(...) 
): 
 
    contents = await file.read() 
 
    array = np.frombuffer( 
        contents, 
        dtype=np.uint8 
    ) 
 
    image = cv2.imdecode( 
        array, 
        cv2.IMREAD_COLOR 
    ) 
 
    if image is None: 
        return { 
            "success": False, 
            "error": "Invalid image" 
        } 
 
    result = pipeline.process( 
        image 
    ) 
 
    response = { 
        "success": result["success"], 
        "plates": [] 
    } 
 
    for plate in result["plates"]: 
 
        response["plates"].append({ 
            "bbox": plate["bbox"], 
            "detection_confidence": 
                plate["detection_confidence"], 
            "raw_ocr": 
                plate["raw_ocr"], 
            "plate": 
                plate["plate"], 
            "format": 
                plate["format"] 
        }) 
 
    return response