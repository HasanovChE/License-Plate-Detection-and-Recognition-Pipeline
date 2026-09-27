from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")
    model.train(
        data="configs/data.yaml",
        epochs=80,
        imgsz=640,
        batch=8,
        workers=2,
        patience=15,
        project="runs",
        name="license_plate_detector",
        pretrained=True,
        verbose=True
    )

if __name__ == "__main__":
    main()