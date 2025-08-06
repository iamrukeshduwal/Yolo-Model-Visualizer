from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from yolo_detection_app.api import update

app = FastAPI()
app.mount("/yolo_detection_app/static", StaticFiles(directory="yolo_detection_app/static"), name="static")
app.include_router(update.router)


