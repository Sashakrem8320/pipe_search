from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO("yolov8s-seg.pt")  # загрузка предобученной модели

    # Запуск обучения
    results = model.train(
        data="YOLO_dataset/data.yaml",
        epochs=1000,
        imgsz=704,
        batch=16,
        device=0,

    )
    metrics = model.val()
    print("Mean Average Precision for boxes:", metrics.box.map)
    print("Mean Average Precision for masks:", metrics.seg.map)
