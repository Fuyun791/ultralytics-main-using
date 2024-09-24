from ultralytics import YOLO

if __name__ == "__main__":
    # Load a model
    # model = YOLO("yolov8n.yaml")  # build a new model from scratch
    model = YOLO("runs/detect/train45/weights/last.pt")
    # 基线map0.5=77.6，map0.5-0.95=48.6  本机实际跑出map0.5=77.8，map0.5-0.95=48.9 results：train46\weights crowedHuman_baseline
    ''' 
    300 epochs completed in 14.780 hours.
    image 1/1 D:\cai\Github\ultralytics-main-using\bus.jpg: 640x480 3 heads, 4 persons, 13.0ms
    Speed: 3.0ms preprocess, 13.0ms inference, 3.0ms postprocess per image at shape (1, 3, 640, 480)

    YOLOv8n summary (fused): 168 layers, 3,006,038 parameters, 0 gradients, 8.1 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 69/69 [01:03<00:00,  1.08it/s]
                   all       4365     181897      0.845      0.685      0.778      0.489
                  head       4355      82364      0.873      0.699      0.786      0.485
                person       4364      99533      0.816      0.671      0.771      0.494
    '''
    # model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

    # Use the model
    model.train(data="CrowedHuman.yaml", epochs=300, batch=32)  # train the model
    metrics = model.val()  # evaluate model performance on the validation set
    results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
    path = model.export(format="onnx")  # export the model to ONNX format
