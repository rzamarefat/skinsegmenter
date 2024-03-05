import cv2
from skinsegmenter import SkinDetection

img = cv2.imread(r"C:\Users\ASUS\Desktop\github_projects\skinsegmenter\demo_images\man_body.png")
skinsegmenter = SkinDetection()
res = skinsegmenter.segment(img)
print(res)