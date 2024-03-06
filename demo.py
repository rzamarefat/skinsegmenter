import cv2
from skinsegmenter import SkinDetection

img = cv2.imread(r"C:\Users\ASUS\Desktop\github_projects\skinsegmenter\demo_images\man_body.png")

skinsegmenter = SkinDetection()
res = skinsegmenter.segment(img)
for index, mask in enumerate(res["instance_masks"]):
    cv2.imwrite(f"instance_mask__{index}.png", mask)
cv2.imwrite("integrated_mask.png", res["integrated_mask"])
cv2.imwrite("visualized_image.png", res["visualized_image"])