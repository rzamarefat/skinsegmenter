import cv2
from skinsegmenter import SkinDetection
from PIL import Image

input_ = "path/to/img.png"
input_ = Image.open(input_)
skinsegmenter = SkinDetection()
res = skinsegmenter.segment(input_)
for index, mask in enumerate(res["instance_masks"]):
    cv2.imwrite(f"instance_mask__{index}.png", mask)
cv2.imwrite("integrated_mask.png", res["integrated_mask"])
cv2.imwrite("visualized_image.png", res["visualized_image"])