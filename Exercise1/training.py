from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo11n.pt")
    results = model.train(data="VisDrone.yaml", epochs=100, imgsz=640)