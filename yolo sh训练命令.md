# yolo sh训练命令

```sh
yolo train model=yolov8n.yaml data=coco128.yaml epochs=100 imgsz=640
```

```sh
# linux系统下实时查看gpu，cpu，内存使用情况
pip install gpustat
gpustat -cp -i 1
```

# 更新本地ultralytics

```sh
pip install -e .
```
