from ultralytics import YOLO

if __name__ == "__main__":
    # Load a model
    model = YOLO("runs/detect/VOC/weights/last.pt")  # build a new model from scratch
    # model = YOLO(
    #     "runs/detect/VOC/weights/best.pt"
    # )  # load a pretrained model (recommended for training)

    # Use the model
    model.train(data="VOC.yaml", epochs=300, resume=True)  # train the model
    metrics = model.val()  # evaluate model performance on the validation set
    results = model(
        "https://ultralytics.com/images/bus.jpg", save=True
    )  # predict on an image
    path = model.export(format="onnx")  # export the model to ONNX format
