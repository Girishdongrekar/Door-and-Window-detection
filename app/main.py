from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import shutil
import uuid
import os
import json  # Import json module to save data

app = FastAPI()
model = YOLO("app/best.pt")  # Load your trained YOLO model

# Define class names from your training config
class_names = ['Door', 'Window']

@app.get("/")
def read_root():
    return {"message": "YOLOv8 Detection API is up and running!"}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    # Save uploaded file temporarily
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

    # ✅ Save results to a JSON file
    os.makedirs("results", exist_ok=True)  # Create 'results/' folder if it doesn't exist
    json_filename = os.path.join("results", f"result_{uuid.uuid4()}.json")
    with open(json_filename, "w") as json_file:
        json.dump({"detections": boxes}, json_file, indent=4)

    # Clean up temp image
    os.remove(temp_filename)

    # Return detection + json filename in response
    return JSONResponse(content={
        "detections": boxes,
        "json_saved_as": json_filename
    })
