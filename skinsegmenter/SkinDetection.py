import copy
import os
import numpy as np
import cv2
from ultralytics import YOLO
import gdown

class SkinDetection:
    def __init__(self, 
                 skin_seg_ckpt_path=os.path.join(os.getcwd(), "weights", "skin_model.pt"), 
                 yolo_seg_ckpt_path=os.path.join(os.getcwd(), "weights", "yolov8l-seg.pt"), 
                 device="cpu",
                 skin_threshold=0.1, 
                 segmentation_confidence=0.5
                 ):
        self._device = device
        self._skin_threshold = skin_threshold
        self._segmentation_confidence = segmentation_confidence

        
        self._skin_seg_ckpt_path = skin_seg_ckpt_path
        self._yolo_seg_ckpt_path = yolo_seg_ckpt_path

        
        os.makedirs(os.path.join(os.getcwd(), "weights"), exist_ok=True)

        if not(os.path.isfile(self._skin_seg_ckpt_path)):
            print("The skin_seg_ckpt_path is set to a non-existent path. Downloading the model...")
            gdown.download(
                        "https://drive.google.com/uc?id=1vyARXVlVpkVth9w73mzHbSE9EtopQMe0", 
                        self._skin_seg_ckpt_path, 
                        quiet=False
                        )
            
        if not(os.path.isfile(self._yolo_seg_ckpt_path)):
            print("The  is yolo_seg_ckpt_path set to a non-existent path. Downloading the model...")
            gdown.download(
                        "https://drive.google.com/uc?id=1kv2gOSlRb4COl9sFYVYvil8DlyTMt4ig", 
                        self._yolo_seg_ckpt_path, 
                        quiet=False
                        
                        )


        self._skin_model=  YOLO(self._skin_seg_ckpt_path)
        self._seg_model = YOLO(self._yolo_seg_ckpt_path)

    def _merge_masks(self, masks, image_shape):
        merged_mask = np.zeros(image_shape, dtype=np.uint8)
        for mask in masks:
            mask = (mask * 255).astype(np.uint8)
            resized_mask = cv2.resize(mask, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_AREA)
            merged_mask += resized_mask
        return (merged_mask > 0).astype(np.uint8)

    def _get_prediction(self, img):
        predskin = self._skin_model.predict(source=img, conf=self._skin_threshold, verbose=False)
        pred = self._seg_model.predict(source=img, conf=self._segmentation_confidence, verbose=False)

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
        integrated_mask = (merged_mask*255).astype(np.uint8)

        cropped_masks = []
        for i in range(len(boxes)):
            if ClassName[pred_cls[i]] in ['person']:
                mask = masks[i, :, :]
                mask = cv2.resize(mask, (dim_asl[1], dim_asl[0]), interpolation=cv2.INTER_AREA)
                cropped_mask = (mask*255).astype(np.uint8)
                cropped_masks.append(cropped_mask)

        return integrated_mask, cropped_masks

    def _apply_overlay(self, original_image, grayscale_masks, color=(0, 255, 0)):
        image_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
    
        # Iterate over grayscale masks and apply overlay
        for mask in grayscale_masks:
            # Convert mask to 3-channel image
            mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            
            # Apply overlay where mask is white
            overlay = np.where(mask_rgb == 255, color, image_bgr)
            
            # Convert overlay to the same data type as original image
            overlay = overlay.astype(image_bgr.dtype)
            
            # Blend overlay with original image
            image_bgr = cv2.addWeighted(overlay, 0.5, image_bgr, 0.5, 0)

        # Convert back to RGB format
        result_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        return result_image
    
    def segment(self, image):
        final_result = {
            "instance_masks": [],
            "integrated_mask": None,
            "visualized_image": None
        }

        integrated_mask, cropped_masks  = self._do_seg(image)
        predicted_mask_instances = []

        if len(cropped_masks) > 0: 
            for idx, mask_crop in enumerate(cropped_masks):
                mask_crop = mask_crop>100
                mask_crop = mask_crop.astype(np.uint8)  # Convert to uint8
                mask_crop = mask_crop * 255
                final_instance_mask = cv2.bitwise_and(integrated_mask, mask_crop)
                predicted_mask_instances.append(final_instance_mask)

            final_result["instance_masks"] = predicted_mask_instances
            final_result["integrated_mask"] = integrated_mask

            final_result["visualized_image"] = self._apply_overlay(image, predicted_mask_instances)

            return final_result
        else:
            return None

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