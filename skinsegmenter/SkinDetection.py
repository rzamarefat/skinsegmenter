import copy
from PIL import Image
import os
import numpy as np
import re
import torch
import cv2
from ultralytics import YOLO
import time

class SkinDetection:
    def __init__(self, device="cpu"):
        self._device = device
        self._skin_model=  YOLO(r"C:\Users\ASUS\Desktop\github_projects\skinseg\weights\SkinModel.pt")
        self._seg_model = YOLO(r"C:\Users\ASUS\Desktop\github_projects\skinseg\weights\yolov8l-seg.pt")

        self.skinThr = 0.1
        self.confidenceSeg = 0.5

    def _merge_masks(self, masks, image_shape):
        merged_mask = np.zeros(image_shape, dtype=np.uint8)
        for mask in masks:
            mask = (mask * 255).astype(np.uint8)
            resized_mask = cv2.resize(mask, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_AREA)
            merged_mask += resized_mask
        return (merged_mask > 0).astype(np.uint8)

    def _get_prediction(self, img):
        predskin = self._skin_model.predict(source=img, conf=self.skinThr)
        pred = self._seg_model.predict(source=img, conf=self.confidenceSeg)

        masksSk = None
        if len(predskin[0]) > 0:
            masksSk = predskin[0].masks.data.cpu().numpy()

        if len(pred[0]) > 0:
            xyxy = pred[0].boxes.xyxy.cpu().numpy()
            cls = pred[0].boxes.cls.cpu().numpy()
            masks = pred[0].masks.data.cpu().numpy()
            ClassName = pred[0].names
        else:
            return [], [], [], [], []

        return masks, xyxy, cls, ClassName, masksSk

    def _do_seg(self, img_real):
        img = copy.deepcopy(img_real)
        dim_asl = img_real.shape[0:2]
        masks, boxes, pred_cls, ClassName, maskSk = self._get_prediction(img_real)
        merged_mask = self._merge_masks(maskSk, dim_asl)
        cropped_masks = []
        for i in range(len(boxes)):
            if ClassName[pred_cls[i]] in ['person']:
                mask = masks[i, :, :]
                mask = cv2.resize(mask, (dim_asl[1], dim_asl[0]), interpolation=cv2.INTER_AREA)
                x1, y1, x2, y2 = boxes[i]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cropped_mask = (mask[y1:y2, x1:x2]*255).astype(np.uint8)
                cropped_img = img[y1:y2, x1:x2]
                skinIm = (merged_mask[y1:y2, x1:x2]*255).astype(np.uint8)
                cropped_masks.append([cropped_mask, cropped_img, skinIm])

        return cropped_masks
    
    def segment(self,image):
        FinalDict = []
        HumanMask  = self._do_seg(image)
        if len(HumanMask) > 0: 
            for idx, (maskCrop, imgCrop, skinCrop) in enumerate(HumanMask):
                imgCrop_real = copy.deepcopy(imgCrop)
                maskCrop = maskCrop>100
                maskCrop = maskCrop.astype(np.uint8)  # Convert to uint8
                maskCrop = maskCrop * 255
                OutputImage = cv2.bitwise_and(skinCrop, maskCrop)

                # erode_kernel = np.ones((9, 9), np.uint8)
                # maskCrop = cv2.erode(maskCrop, erode_kernel, iterations=1)
                NumHuman = self.count_pixels_with_value(maskCrop,255)
                NumSkin  = self.count_pixels_with_value(OutputImage, 255)
                if (NumSkin!=0 and NumHuman!=0):
                    PersentSkin = NumSkin / NumHuman
                else:
                    PersentSkin = 0

                # FinalDict.append({"ImgCrop":imgCrop_real, "PersentSkin":PersentSkin,"SkinMask":OutputImage})
                FinalDict.append({"realImgCrop": imgCrop_real, "skinPercent": np.round(100*PersentSkin), "skinMask": OutputImage})

                
                # mask_3channel = np.repeat(OutputImage[:, :, np.newaxis], 3, axis=2)
                # output_ = cv2.bitwise_and(imgCrop_real, mask_3channel)
                # cv2.imwrite("OutImage.jpg",output_)
                # print("PersentSkin: %", np.round(100*PersentSkin))

            return FinalDict
        else:
            return [{"realImgCrop": None, "skinPercent": None, "skinMask": None}]

    def count_pixels_with_value(self,image, value):
        count = 0
        for row in image:
            for pixel in row:
                if pixel == value:
                    count += 1
        return count

if __name__ =="__main__":
    dir_ = os.getcwd()
    self_ = SkinDetectionClass()## 384
    img_path = dir_ + "/Im/3.png"
    im = cv2.imread(img_path)
    self_.inferenceModel(Im = im)