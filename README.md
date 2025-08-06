# YOLO Model Visualizer

A FastAPI-based web app to run YOLO (v5 GitHub, v8, v11) inference on images with support for **normal**, **segmentation**, **OBB**, and **classification** models.  

## How to Run
```bash
git clone https://github.com/iamrukeshduwal/Yolo-Model-Visualizer.git
cd Yolo-Model-Visualizer
git clone https://github.com/ultralytics/yolov5.git   # optional for YOLOv5 models
pip install -r requirements.txt
uvicorn yolo_detection_app.main:app --reload
Open http://127.0.0.1:8000 in your browser.
