import cv2
from skinsegmenter import SkinDetection

img = cv2.imread(r"C:\Users\ASUS\Desktop\github_projects\skinsegmenter\demo_images\two_men.jpg")
skinsegmenter = SkinDetection()
res = skinsegmenter.segment(img)
print(res)