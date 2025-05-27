from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO
import shutil
import uuid
import os
import json

app = FastAPI()

# Load YOLOv8 model once globally
model = YOLO("app/best.pt")
class_names = ['Door', 'Window']

# Create folders if not exist
os.makedirs("results", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Set up templates and static file serving
templates = Jinja2Templates(directory="templates")
app.mount("/results", StaticFiles(directory="results"), name="results")

@app.get("/", response_class=HTMLResponse)
def redirect_to_upload(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@app.get("/upload", response_class=HTMLResponse)
def upload_page(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    # Save uploaded image
    file_ext = file.filename.split('.')[-1]
    temp_filename = f"temp_{uuid.uuid4()}.{file_ext}"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run detection
    results = model(temp_filename)
    boxes = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            boxes.append({
                "bbox": [round(v, 2) for v in box.xyxy[0].tolist()],
                "class_id": class_id,
                "class_name": class_names[class_id],
                "confidence": round(float(box.conf[0]), 2)
            })

    # Save detection result as JSON
    json_name = f"result_{uuid.uuid4()}.json"
    json_path = os.path.join("results", json_name)
    with open(json_path, "w") as json_file:
        json.dump({"detections": boxes}, json_file, indent=4)

    # Save image with bounding boxes
    detected_img_name = f"detected_{uuid.uuid4()}.jpg"
    detected_img_path = os.path.join("results", detected_img_name)
    results[0].save(filename=detected_img_path)

    # Remove temp image
    os.remove(temp_filename)

    base_url = "https://door-and-window-detection-uing-yolov8.onrender.com"

    return JSONResponse(content={
        "detections": boxes,
        "json_result_url": f"{base_url}/results/{json_name}",
        "detected_image_url": f"{base_url}/results/{detected_img_name}"
    })

@app.on_event("startup")
def print_links():
    print("\n🚀 YOLOv8 Detection API is live!")
    print("📤 Upload Page: http://127.0.0.1:8000/upload\n")
