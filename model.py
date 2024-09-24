from ultralytics import YOLO

"""
YOLOv8n_EMA summary: 249 layers, 3,157,242 parameters, 3,157,226 gradients, 8.9 GFLOPs

YOLOv8n summary: 225 layers, 3,157,200 parameters, 3,157,184 gradients, 8.9 GFLOPs

YOLOv8n_SPD summary: 231 layers, 3,541,200 parameters, 3,541,184 gradients, 13.2 GFLOPs
"""

if __name__ == "__main__":
    # Load a model
    model = YOLO("yolov8n_SPD.yaml")  # build a new model from scratch
    # model = YOLO(
    #     "runs/detect/train45/weights/last.pt"
    # )  # 应该map0.5要跑到77.6，map0.5-0.95=48.6
    # # model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
    # print(model)
    model.train(data="coco128.yaml", epochs=1, batch=1)  # train the model
    # model.info()  # train the model
