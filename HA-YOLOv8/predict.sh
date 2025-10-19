model=/home/zt/Project/Detect_Segment/ultralytics/runs/segment/ori_yolov8l_imgsz1280_train/weights/best.pt
source=/home/zt/temp/2.jpg

yolo predict model=$model source=$source imgsz=1280