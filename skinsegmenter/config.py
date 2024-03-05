import os

cwd = os.getcwd()

PATH_TO_YOLO_FACE_CKPT = os.path.join(cwd, "YOLOv8_FACE.pt")
YOLO_FACE_CONFIDENCE_THRESHOLD = 0.7


MARGIN_FOR_FACE = 10
