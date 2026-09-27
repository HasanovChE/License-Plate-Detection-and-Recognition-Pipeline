# License Plate Detection and Recognition Pipeline

End-to-end license plate detection and recognition system using YOLOv8, OpenCV and OCR.

The system detects license plates from vehicle images, crops the detected regions, performs OCR, cleans the recognized text, validates the plate format, evaluates detection and OCR performance, and optionally records recognized plates in a SQLite parking log.

## Features

- YOLOv8 license plate detector
- Custom license plate dataset
- Train/validation/test split
- IoU-based detection evaluation
- Mean IoU
- 20+ held-out test images
- Multiple vehicle support
- Zero-detection handling
- Automatic plate cropping
- Crop padding
- Tesseract OCR
- Raw OCR output
- OCR text cleaning
- Rule-based character correction
- Exact-match accuracy
- Character Error Rate
- End-to-end accuracy
- Failure case analysis
- Blur handling
- Angled plate handling
- Partial occlusion handling
- No manual cropping
- EasyOCR
- Tesseract vs EasyOCR comparison
- Webcam recognition
- Dashcam/video recognition
- Live bounding boxes
- Plate format validation
- Multi-country plate-format classifier
- SQLite parking log
- Timestamped detections
- FastAPI inference API
- Health endpoint
- Automated tests
- Production-oriented project structure

## Architecture

```
Input Image
    |
YOLOv8 Detector
    |
Bounding Boxes
    |
Padding + Crop
    |
OCR Preprocessing
    |
    +----> Tesseract
    |
    +----> EasyOCR
    |
Text Cleaning
    |
Regex Validation
    |
Country / Format Classification
    |
Parking Log / API Response
```

## Hardware

Development configuration:
- Python 3.10+
- NVIDIA GPU preferred for training
- 8 GB RAM for lightweight local inference
- 256 GB SSD minimum

An CPU can run the inference pipeline, OpenCV, Tesseract, SQLite and FastAPI, but YOLO training is better performed on Google Colab GPU.

The trained `best.pt` model then be copied back to the local machine for inference.

## Dataset

121 labeled images

The dataset contains:
- frontal plates
- angled plates
- distant vehicles
- close vehicles
- blurred plates
- partially occluded plates
- multiple vehicles
- different lighting conditions

The detector class is:
```
0 = license_plate
```

### Dataset structure

```
data/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
│
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
│
└── metadata/
    └── plates.csv
```

## Installation

```bash
git clone <repository-url>
cd license-plate-recognition
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### Tesseract

Install Tesseract OCR separately on Windows.

Verify:
```bash
tesseract --version
```

Tesseract is not in PATH, configure the executable path in the application.

Typical path:
```
C:\Program Files\Tesseract-OCR\tesseract.exe
```

## Training

Using the YOLO dataset configuration:
```
configs/data.yaml
```

Run:
```bash
python scripts/train.py
```

The best model is:
```
runs/license_plate_detector/weights/best.pt
```

Copy it to:
```
models/best.pt
```

## Image prediction

```bash
python scripts/predict.py --image test.jpg
```

Pipeline:
```
image
-> YOLO
-> bbox
-> crop
-> preprocessing
-> OCR
-> cleaning
-> validation
-> result
```

## Webcam

```bash
python scripts/webcam.py
```

`q` to exit.

## Video

Using the video processing script with a dashcam or recorded vehicle video.

The system draws:
- bounding boxes
- recognized plate text
- detection confidence

## API

Start:
```bash
uvicorn api.main:app --reload
```

Open:
```
http://127.0.0.1:8000/docs
```

Health: `GET /health`
Prediction: `POST /predict`

### Example API response

```json
{
  "success": true,
  "plates": [
    {
      "bbox": [240, 180, 390, 225],
      "detection_confidence": 0.91,
      "raw_ocr": "99 AB 123",
      "plate": "99AB123",
      "format": {
        "country": "AZERBAIJAN",
        "confidence": 1.0
      }
    }
  ]
}
```

## Evaluation

Run:
```bash
python scripts/evaluate.py
```

The evaluation includes:

**Detection**
- IoU
- mean IoU
- IoU >= 0.50
- detection recall

**OCR**
- exact match
- Character Error Rate

**End-to-end**
```
Image
-> Detection
-> Crop
-> OCR
-> Cleaning
-> Ground Truth Comparison
```

Results are saved to:
```
outputs/evaluation/results.csv
```

## OCR comparison

The same test crops are sent to both Tesseract and EasyOCR. Their exact-match accuracy and CER are compared.

## Failure analysis

Three failure cases:

**Blur** — Motion blur can make individual characters difficult to separate.

**Perspective** — Angled plates can distort character geometry.

**Occlusion** — A partially hidden plate can prevent the OCR engine from reading all characters.

Failure examples are saved in:
```
outputs/failures/
```

## Parking Log

Recognized plates stored in SQLite:
```
parking.db
```

Stored information:
- plate
- country
- confidence
- timestamp
- image path

Repeated detections filtered using a configurable cooldown period.

## Tests

Run:
```bash
pytest
```

Tests cover:
- IoU calculation
- crop extraction
- text cleaning
- pipeline behavior

## Production considerations

For production deployment:
- using a fixed dependency lock file
- validation uploaded images
- limited upload size
- using structured logging
- keeping model paths in configuration
- separation detector and OCR components
- exposing health checks
- using SQLite for small deployments and PostgreSQL for multi-user deployments
- considering ONNX export for deployment optimization
- using GPU inference when available
- addding monitoring for detection confidence and OCR failures

## Model export

Ultralytics supports model export for deployment formats such as ONNX. The exported model can be used when the deployment environment does not need the full Python training stack.

## Project workflow

1. Collect images
2. Annotate plates
3. Convert/verify YOLO labels
4. Split dataset
5. Train YOLOv8
6. Select best.pt
7. Run held-out evaluation
8. Calculate IoU
9. Detect plates
10. Crop automatically
11. Run OCR
12. Clean text
13. Validate format
14. Calculate exact match
15. Calculate CER
16. Compare Tesseract and EasyOCR
17. Analyze failures
18. Test webcam
19. Test dashcam video
20. Store detections
21. Expose API
22. Test unseen images

## Important

Detection and OCR are evaluated separately.

A correct final result requires both:
```
correct detection
+
correct OCR
=
correct end-to-end recognition
```

## License

Review the licenses of the selected Ultralytics model, dataset, OCR engine and any third-party assets before commercial deployment.