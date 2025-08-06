@echo off
cd /d "D:\CrimsonTech2025\New folder\yolo_detection\yolo_detection"
"C:\Users\ACER\AppData\Local\Programs\Python\Python310\python.exe" -m uvicorn yolo_detection_app.main:app --reload
pause
