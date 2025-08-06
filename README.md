# YOLO Detection Viewer

This is a simple web-based application built using FastAPI that allows users to run YOLO (v5, v8, classification, segmentation, OBB) model inference on a folder of images.

It provides an easy-to-use interface to:
- Load a folder of images
- Select a YOLO model (`.pt` file)
- Choose model type (`normal`, `segmentation`, `obb`, or `classification`)
- Adjust confidence and IoU thresholds using sliders
- Navigate through images and view detections

The UI is built with Bootstrap and includes keyboard shortcuts, live preview, and automatic result updates.

---

## How to Run

1. Clone the repository

```bash
git clone https://github.com/iamrukeshduwal/Yolo-Model-Visualizer.git
cd Yolo-Model-Visualizer

pip install -r requirements.txt
uvicorn yolo_detection_app.main:app --reload
http://127.0.0.1:8000
