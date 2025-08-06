import os
import sys
import cv2
import torch
import numpy as np
from collections import Counter
from datetime import datetime
from ultralytics import YOLO

# ─── Add YOLOv5 GitHub path ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YOLOV5_PATH = os.path.join(BASE_DIR, "yolov5")
if YOLOV5_PATH not in sys.path:
    sys.path.append(YOLOV5_PATH)

try:
    from detect import run as yolov5_run
except ImportError:
    yolov5_run = None


def is_yolov5_model(path):
    """Check if this is a YOLOv5 GitHub model by scanning for 'models.yolo' in pickle bytes."""
    try:
        with open(path, "rb") as f:
            return b"models.yolo" in f.read()
    except Exception:
        return False


def get_random_color(seed):
    np.random.seed(seed)
    return tuple(np.random.randint(0, 255, 3).tolist())


def run_detection(image_path, model_path, conf=0.25, iou=0.45, model_type="normal", engine="auto"):
    """
    Runs detection for YOLOv5 GitHub or Ultralytics YOLOv8/YOLOv11 models.
    Returns:
        ui_image_path (str): Path to annotated image in static folder.
        class_counts (dict): Class name → count.
    """

    # ─────────────────────────────
    # YOLOv5 GitHub route
    # ─────────────────────────────
    if is_yolov5_model(model_path):
        if yolov5_run is None:
            raise ImportError("YOLOv5 detect.py not found in cloned repo path.")

        print(f"⚠ Detected YOLOv5 GitHub model at {model_path} – running detect.py")

        save_dir = os.path.join("yolo_detection_app", "static")
        os.makedirs(save_dir, exist_ok=True)
        exp_name = "det_yolov5"

        yolov5_run(
            weights=model_path,
            source=image_path,
            conf_thres=conf,
            iou_thres=iou,
            save_txt=False,
            save_conf=False,
            save_crop=False,
            nosave=False,
            project=save_dir,
            name=exp_name,
            exist_ok=True
        )

        # Find latest annotated image
        exp_path = os.path.join(save_dir, exp_name)
        result_images = [os.path.join(exp_path, f) for f in os.listdir(exp_path)
                         if f.lower().endswith(('.jpg', '.png'))]
        if not result_images:
            raise FileNotFoundError("No output images found from YOLOv5 detection.")
        latest_img = max(result_images, key=os.path.getctime)

        # Copy to root static folder for UI
        ui_image_path = os.path.join(save_dir, f"det_{datetime.now().strftime('%H%M%S_%f')}.jpg")
        import shutil
        shutil.copy(latest_img, ui_image_path)

        # Optional: parse CSV if detect.py writes one
        csv_file = os.path.join(exp_path, "predictions.csv")
        class_counts = {}
        if os.path.exists(csv_file):
            import csv
            with open(csv_file, newline="") as f:
                reader = csv.DictReader(f)
                labels = [row["Prediction"] for row in reader]
                class_counts = dict(Counter(labels))

        return ui_image_path, class_counts

    # ─────────────────────────────
    # Ultralytics YOLOv8 / YOLOv11 route
    # ─────────────────────────────
    print(f"ℹ Using Ultralytics YOLO for {model_path}")
    image = cv2.imread(image_path)
    annotated = image.copy()
    class_counts = {}

    model = YOLO(model_path)
    names = model.names
    results = model(image, conf=conf, iou=iou)[0]
    class_ids = []

    if model_type == "classification":
        cls_id = int(results.probs.top1)
        score = float(results.probs.top1conf)
        label = names[cls_id]
        class_counts = {label: score}
        cv2.putText(annotated, f"{label} ({score:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    elif model_type == "normal":
        for box, cls, score in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
            x1, y1, x2, y2 = map(int, box)
            cls_id = int(cls)
            label = names[cls_id]
            class_ids.append(cls_id)
            color = get_random_color(cls_id)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f"{label} {score:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    elif model_type == "segmentation" and hasattr(results, 'masks') and results.masks is not None:
        for mask, cls, score in zip(results.masks.data, results.boxes.cls, results.boxes.conf):
            cls_id = int(cls)
            label = names[cls_id]
            class_ids.append(cls_id)
            mask = mask.cpu().numpy().astype("uint8") * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            color = get_random_color(cls_id)
            cv2.drawContours(annotated, contours, -1, color, 2)
            if contours:
                x, y = contours[0][0][0]
                cv2.putText(annotated, f"{label} {score:.2f}", (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    elif model_type == "obb" and hasattr(results, 'obb') and results.obb is not None:
        for polygon, cls, score in zip(results.obb.xyxyxyxy, results.obb.cls, results.obb.conf):
            cls_id = int(cls)
            label = names[cls_id]
            class_ids.append(cls_id)
            box = polygon.cpu().numpy().astype(int).reshape(-1, 1, 2)
            color = get_random_color(cls_id)
            cv2.polylines(annotated, [box], isClosed=True, color=color, thickness=2)
            cx, cy = box[0][0]
            cv2.putText(annotated, f"{label} {score:.2f}", (cx, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if model_type != "classification":
        class_counts = Counter([names[i] for i in class_ids])

    # Save annotated image for UI
    save_dir = os.path.join("yolo_detection_app", "static")
    os.makedirs(save_dir, exist_ok=True)
    ui_image_path = os.path.join(save_dir, f"det_{datetime.now().strftime('%H%M%S_%f')}.jpg")
    cv2.imwrite(ui_image_path, annotated)

    return ui_image_path, class_counts
