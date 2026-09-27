"""
Evaluate the detection + OCR pipeline against a labeled test set.

Expects:
  - data/images/test/*.jpg | *.png          (test images)
  - data/labels/test/<same-name>.txt        (YOLO format: class x_center y_center width height, normalized)
  - data/metadata/plates.csv                (columns: image, plate  OR  image, plate_id, plate)
  - models/best.pt

Outputs:
  - outputs/evaluation/results.csv
  - a printed summary report
"""

import argparse
from pathlib import Path

import cv2
import pandas as pd

from src.pipeline.pipeline import LicensePlatePipeline
from src.evaluation.detection_metrics import calculate_iou

IOU_MATCH_THRESHOLD = 0.5
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def normalize_plate_text(text):
    return (
        str(text).upper()
        .replace(" ", "")
        .replace("-", "")
        .replace(".", "")
    )


def exact_match(predicted, truth):
    return normalize_plate_text(predicted) == normalize_plate_text(truth)


def character_error_rate(predicted, truth):
    predicted = normalize_plate_text(predicted)
    truth = normalize_plate_text(truth)

    m, n = len(truth), len(predicted)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if truth[i - 1] == predicted[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    if m == 0:
        return 0.0 if n == 0 else 1.0
    return dp[m][n] / m


def load_yolo_labels(label_path, image_width, image_height):
    """Read a YOLO-format label file, return pixel-space [x1, y1, x2, y2] boxes."""
    boxes = []
    if not label_path.exists():
        return boxes

    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            _, xc, yc, w, h = parts
            xc, yc, w, h = float(xc), float(yc), float(w), float(h)

            box_w = w * image_width
            box_h = h * image_height
            cx = xc * image_width
            cy = yc * image_height

            x1 = cx - box_w / 2
            y1 = cy - box_h / 2
            x2 = cx + box_w / 2
            y2 = cy + box_h / 2

            boxes.append([int(x1), int(y1), int(x2), int(y2)])
    return boxes


def load_plate_ground_truth(csv_path):
    """Return {image_name: [plate_text, ...]}, ordered by plate_id when present."""
    gt = {}
    if not csv_path.exists():
        return gt

    df = pd.read_csv(csv_path)
    if "plate_id" in df.columns:
        df = df.sort_values(["image", "plate_id"])

    for _, row in df.iterrows():
        gt.setdefault(row["image"], []).append(str(row["plate"]))

    return gt


def match_predictions_to_ground_truth(pred_boxes, gt_boxes):
    """Greedy IoU matching: returns list of (pred_idx, gt_idx, iou), highest IoU first."""
    pairs = []
    for pi, pbox in enumerate(pred_boxes):
        for gi, gbox in enumerate(gt_boxes):
            pairs.append((calculate_iou(pbox, gbox), pi, gi))

    pairs.sort(key=lambda x: x[0], reverse=True)

    matched_pred, matched_gt = set(), set()
    matches = []
    for iou, pi, gi in pairs:
        if pi in matched_pred or gi in matched_gt:
            continue
        matched_pred.add(pi)
        matched_gt.add(gi)
        matches.append((pi, gi, iou))

    return matches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/best.pt")
    parser.add_argument("--test-dir", default="data/images/test")
    parser.add_argument("--labels-dir", default="data/labels/test")
    parser.add_argument("--plates-csv", default="data/metadata/plates.csv")
    parser.add_argument("--output", default="outputs/evaluation/results.csv")
    args = parser.parse_args()

    test_dir = Path(args.test_dir)
    labels_dir = Path(args.labels_dir)

    image_paths = sorted(
        p for p in test_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        print(f"No test images found in {test_dir}")
        return

    plate_gt = load_plate_ground_truth(Path(args.plates_csv))
    pipeline = LicensePlatePipeline(model_path=args.model)

    rows = []
    ious = []

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        height, width = image.shape[:2]
        gt_boxes = load_yolo_labels(labels_dir / f"{image_path.stem}.txt", width, height)
        gt_plates = plate_gt.get(image_path.name, [])

        result = pipeline.process(image)
        pred_boxes = [p["bbox"] for p in result["plates"]]

        matches = match_predictions_to_ground_truth(pred_boxes, gt_boxes)

        if not matches:
            rows.append({
                "image": image_path.name,
                "gt_box": gt_boxes[0] if gt_boxes else None,
                "pred_box": pred_boxes[0] if pred_boxes else None,
                "iou": 0.0,
                "gt_plate": gt_plates[0] if gt_plates else None,
                "raw_ocr": result["plates"][0]["raw_ocr"] if result["plates"] else None,
                "cleaned_plate": result["plates"][0]["plate"] if result["plates"] else None,
                "exact_match": False,
                "cer": 1.0,
                "country": result["plates"][0]["format"]["country"] if result["plates"] else None,
            })
            continue

        for pi, gi, iou in matches:
            ious.append(iou)
            pred = result["plates"][pi]
            gt_plate = gt_plates[gi] if gi < len(gt_plates) else None

            is_match = exact_match(pred["plate"], gt_plate) if gt_plate else False
            cer = character_error_rate(pred["plate"], gt_plate) if gt_plate else None

            rows.append({
                "image": image_path.name,
                "gt_box": gt_boxes[gi],
                "pred_box": pred["bbox"],
                "iou": round(iou, 3),
                "gt_plate": gt_plate,
                "raw_ocr": pred["raw_ocr"],
                "cleaned_plate": pred["plate"],
                "exact_match": is_match,
                "cer": round(cer, 3) if cer is not None else None,
                "country": pred["format"]["country"],
            })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)

    mean_iou = sum(ious) / len(ious) if ious else 0.0
    recall_50 = sum(1 for i in ious if i >= IOU_MATCH_THRESHOLD)
    exact_matches = df["exact_match"].sum() if "exact_match" in df else 0
    cer_values = df["cer"].dropna() if "cer" in df else []
    mean_cer = cer_values.mean() if len(cer_values) else None
    end_to_end_correct = sum(
        1 for _, r in df.iterrows()
        if r.get("exact_match") and (r.get("iou") or 0) >= IOU_MATCH_THRESHOLD
    )

    print("=" * 40)
    print("LICENSE PLATE PIPELINE EVALUATION")
    print("=" * 40)
    print(f"\nTest images: {len(image_paths)}")
    print("\nDetection")
    print("-" * 40)
    print(f"Mean IoU: {mean_iou:.2f}")
    print(f"IoU >= 0.50: {recall_50}/{len(ious)}")
    print(f"Detection recall: {(recall_50 / len(ious) * 100) if ious else 0:.0f}%")
    print("\nOCR")
    print("-" * 40)
    print(f"Exact Match: {(exact_matches / len(df) * 100) if len(df) else 0:.0f}%")
    print(f"Mean CER: {mean_cer:.2f}" if mean_cer is not None else "Mean CER: n/a")
    print("\nEnd-to-End")
    print("-" * 40)
    print(f"Correct: {end_to_end_correct}/{len(df)}")
    print(f"Accuracy: {(end_to_end_correct / len(df) * 100) if len(df) else 0:.0f}%")
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
