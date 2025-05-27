# train.py

from ultralytics import YOLO

def main():
    model = YOLO('yolov8n.pt')  # You can switch to yolov8s.pt for more accuracy
    
    model.train(
    data=r'C:\Users\balak\Desktop\project1\data.yaml',
    epochs=100,
    imgsz=640,
    project='training_logs',
    name='yolov8-door',
    exist_ok=True
)


if __name__ == '__main__':
    main()
