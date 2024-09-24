from ultralytics import YOLO

if __name__ == "__main__":
    # Load a model
    """
    YOLOv8n summary: 225 layers, 3,157,200 parameters, 3,157,184 gradients, 8.9 GFLOPs
    YOLOv8n_SPD summary: 231 layers, 3,541,200 parameters, 3,541,184 gradients, 13.2 GFLOPs

    ---网上效果 coco
    YOLOv8_SPD summary (fused): 174 layers, 3451283 parameters, 0 gradients, 50.9 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 37/37 [00:22<00:00,  1.68it/s]
                   all        582       6970      0.828      0.742      0.802      0.416
    ---
    """
    model = YOLO("yolov8n_SPD.yaml")  # build a new model from scratch
    # model = YOLO(
    #     "runs/detect/train45/weights/last.pt"
    # )  # 基线map0.5=77.6，map0.5-0.95=48.6  本机跑出map0.5=77.6，map0.5-0.95=48.6
    # model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

    # Use the model
    model.train(data="CrowedHuman.yaml", epochs=300, batch=32)  # train the model
    metrics = model.val()  # evaluate model performance on the validation set
    # results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
    path = model.export(format="onnx")  # export the model to ONNX format
