import os, json
from typing import Optional
from fastapi import APIRouter, Request, Form
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

from yolo_detection_app.detector import run_detection

router = APIRouter()
templates = Jinja2Templates(directory="yolo_detection_app/templates")

CONFIG_PATH = "yolo_detection_app/thresholds.json"


def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            d = json.load(f)
            return d.get("conf", 0.25), d.get("iou", 0.45)
    return 0.25, 0.45


def save_config(conf, iou):
    with open(CONFIG_PATH, "w") as f:
        json.dump({"conf": conf, "iou": iou}, f)


current_conf, current_iou = load_config()

STATE = {
    "folder": "",
    "model_path": "",
    "model_type": "normal",
    "engine": "auto",
    "index": 0,
    "files": [],
    "result_img": None,
    "class_counts": {}
}


@router.get("/")
def home(request: Request,
         index: int = 0,
         folder: Optional[str] = None,
         model_path: Optional[str] = None,
         model_type: str = "normal",
         engine: str = "auto",
         conf: float = current_conf,
         iou: float = current_iou):

    if folder and model_path and os.path.isdir(folder):
        files = sorted([f for f in os.listdir(folder)
                        if f.lower().endswith((".jpg", ".png", ".jpeg",".bmp"))])
        if files:
            index = max(0, min(index, len(files) - 1))
            image_path = os.path.join(folder, files[index])
            result_img, class_counts = run_detection(image_path, model_path, conf, iou, model_type, engine)

            STATE.update({
                "folder": folder,
                "files": files,
                "index": index,
                "model_path": model_path,
                "model_type": model_type,
                "engine": engine,
                "result_img": "/" + result_img.replace("\\", "/"),
                "class_counts": class_counts
            })

    return templates.TemplateResponse("index.html", {
    "request": request,
    "folder": STATE.get("folder", ""),
    "model_path": STATE.get("model_path", ""),
    "model_type": STATE.get("model_type", "normal"),
    "engine": STATE.get("engine", "auto"),
    "index": STATE.get("index", 0),
    "total": len(STATE.get("files", [])),
    "files": STATE.get("files", []),  # ✅ Pass files list here
    "result_img": STATE.get("result_img", None),
    "class_counts": STATE.get("class_counts", {}),
    "conf": current_conf,
    "iou": current_iou,
    "message": None
})


@router.post("/set-folder")
def set_folder(
    request: Request,
    folder_path: str = Form(...),
    model_path: str = Form(...),
    model_type: str = Form(...),
    engine: str = Form(...)
):
    global current_conf, current_iou

    if not os.path.isdir(folder_path):
        return templates.TemplateResponse("index.html", {
            "request": request,
            "message": "Invalid folder path!"
        })

    files = sorted([f for f in os.listdir(folder_path)
                    if f.lower().endswith((".jpg", ".png", ".jpeg",".bmp"))])
    if not files:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "message": "No image files found!"
        })

    image_path = os.path.join(folder_path, files[0])
    result_img, class_counts = run_detection(image_path, model_path, current_conf, current_iou, model_type, engine)

    STATE.update({
        "folder": folder_path,
        "files": files,
        "index": 0,
        "model_path": model_path,
        "model_type": model_type,
        "engine": engine,
        "result_img": "/" + result_img.replace("\\", "/"),
        "class_counts": class_counts
    })

    return templates.TemplateResponse("index.html", {
        "request": request,
        "folder": folder_path,
        "model_path": model_path,
        "model_type": model_type,
        "engine": engine,
        "index": 0,
        "total": len(files),
        "result_img": "/" + result_img.replace("\\", "/"),
        "class_counts": class_counts,
        "conf": current_conf,
        "iou": current_iou,
        "message": None
    })




@router.get("/live_update")
def live_update(index: int, folder: str, model_path: str,
                model_type: str = "normal", engine: str = "auto",
                conf: float = 0.25, iou: float = 0.45):
    global current_conf, current_iou
    current_conf, current_iou = conf, iou
    save_config(conf, iou)

    files = sorted([f for f in os.listdir(folder)
                    if f.lower().endswith((".jpg", ".png", ".jpeg",".bmp"))])
    index = max(0, min(index, len(files) - 1))

    image_path = os.path.join(folder, files[index])
    result_img, class_counts = run_detection(image_path, model_path, conf, iou, model_type, engine)

    STATE.update({
        "folder": folder,
        "files": files,
        "index": index,
        "model_path": model_path,
        "model_type": model_type,
        "engine": engine,
        "result_img": "/" + result_img.replace("\\", "/"),
        "class_counts": class_counts
    })

    return JSONResponse({
        "result_img": STATE["result_img"],
        "class_counts": class_counts
    })
