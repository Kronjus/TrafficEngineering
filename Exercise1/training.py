from ultralytics import YOLO

# train 6 = good
if __name__ == "__main__":
    model = YOLO("yolo11n.pt")
    results = model.train(data="VisDrone.yaml", epochs=100, imgsz=960, patience=10, batch=8, mosaic=1.0, mixup=0.15,
                          copy_paste=0.1)
