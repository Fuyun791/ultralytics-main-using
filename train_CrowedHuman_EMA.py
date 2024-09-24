from ultralytics import YOLO

if __name__ == "__main__":
    # Load a model
    """
    YOLOv8n_EMA summary: 249 layers, 3,157,242 parameters, 3,157,226 gradients, 8.9 GFLOPs

    YOLOv8n summary: 225 layers, 3,157,200 parameters, 3,157,184 gradients, 8.9 GFLOPs

    baseline情况下基线map0.5=77.6，map0.5-0.95=48.6  本机跑出map0.5=77.8，map0.5-0.95=48.9
    这个的轮数很可能不够再加200epoch
    300 epochs completed in 18.013 hours.
    YOLOv8n_EMA summary (fused): 192 layers, 3,006,080 parameters, 0 gradients, 8.1 GFLOPs
    Results saved to d:\cai\Github\ultralytics-main-using\runs\detect\train47
    val: Scanning C:\Users\admin\c_datasets\CrowdedHuman\labels\val.cache... 4365 images, 1 backgrounds, 0 corrupt: 100%|██████████| 4365/4365 [00:00<?, ?it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 137/137 [01:09<00:00,  1.98it/s]
                   all       4365     181897      0.846      0.685      0.778      0.489
    应该能跑到YOLOv8n-EMA map50=79.1 map50-95=49.7
    """
    # model = YOLO("yolov8n_EMA.yaml")  # build a new model from scratch
    model = YOLO(
        "runs/detect/train47/weights/best.pt"
    )  #
    # model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

    # Use the model
    model.train(data="CrowedHuman.yaml", epochs=200, batch=32)  # train the model
    metrics = model.val()  # evaluate model performance on the validation set
    # results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
    path = model.export(format="onnx")  # export the model to ONNX format
