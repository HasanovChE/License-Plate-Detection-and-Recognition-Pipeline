import argparse
import cv2
from src.pipeline.pipeline import LicensePlatePipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        required=True
    )
    parser.add_argument(
        "--model",
        default="models/best.pt"
    )
    args = parser.parse_args()
    
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(
            args.image
        )
        
    pipeline = LicensePlatePipeline(
        model_path=args.model
    )
    result = pipeline.process(image)
    for p in result["plates"]:
        p.pop("crop", None)
    print(result)

if __name__ == "__main__":
    main()