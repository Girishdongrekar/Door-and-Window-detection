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

# Load YOLOv8 model
model = YOLO("app/best.pt")  # Adjust path if needed
class_names = ['Door', 'Window']

# Create required folders
os.makedirs("results", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Set up templates and static file serving
templates = Jinja2Templates(directory="templates")
app.mount("/results", StaticFiles(directory="results"), name="results")


@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})


@app.get("/upload", response_class=HTMLResponse)
def upload_page(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    # Save uploaded image temporarily
    file_ext = file.filename.split('.')[-1]
    temp_filename = f"temp_{uuid.uuid4()}.{file_ext}"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run YOLOv8 inference
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

    # Save detection results to JSON
    json_name = f"result_{uuid.uuid4()}.json"
    json_path = os.path.join("results", json_name)
    with open(json_path, "w") as json_file:
        json.dump({"detections": boxes}, json_file, indent=4)

    # Save rendered image with detections
    detected_img_name = f"detected_{uuid.uuid4()}.jpg"
    detected_img_path = os.path.join("results", detected_img_name)
    results[0].save(filename=detected_img_path)

    # Remove temporary image
    os.remove(temp_filename)

    # Return JSON and image URL
    return JSONResponse(content={
        "detections": boxes,
        "json_result_url": f"/results/{json_name}",
        "detected_image_url": f"/results/{detected_img_name}"
    })

@app.on_event("startup")
def print_useful_links():
    print("\n🚀 YOLOv8 Detection API is live!")
    print("📤 Upload Page:      http://127.0.0.1:8000/upload")
    print("📚 Swagger Docs:     http://127.0.0.1:8000/docs")
    print("🏠 Root Endpoint:    http://127.0.0.1:8000/\n")