from ultralytics import YOLO

if __name__ == "__main__":
    # Load a model
    model = YOLO("yolov8s.pt")  # load an official model
    # Validate the model
    metrics = model.val(
        data="ultralytics/datasets/coco.yaml", iou=0.7, conf=0.001, half=False, device=0
    )  # no arguments needed, dataset and settings remembered
    metrics = model.val()
    # metrics.box.map    # map50-95
    # metrics.box.map50  # map50
    # metrics.box.map75  # map75
    # metrics.box.maps   # a list contains map50-95 of each category
