from ultralytics import YOLO

def main():
    model = YOLO("models/best.pt")
    
    model.export(format="onnx")
    print("Model successfully exported to ONNX format!")

if __name__ == "__main__":
    main()