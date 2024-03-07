<div align="center">
  <h2>
    Skin Segmentation
  </h2>
  <h3>
    An easy to use and packaged tool for skin segmentation   
  </h3>
    <a href="https://badge.fury.io/py/yolotext"><img src="https://badge.fury.io/py/yolotext.svg" alt="pypi version"></a>
</div>

### Sample Results
![visualized_image](https://github.com/rzamarefat/skinsegmenter/assets/79300456/887c7f5f-f9d0-4ac1-8346-b87c6c2701c5)
![visualized_image](https://github.com/rzamarefat/skinsegmenter/assets/79300456/c8b38beb-f80b-4b25-86ab-ed6734da4e7f)

### Installation

```
pip install 
```


### Usage
```python
import cv2
from skinsegmenter import SkinDetection
from PIL import Image

# option #1
input_ = "path/to/img.png"
input_ = Image.open(input_)

# option #2
input_ = "path/to/img.png"
input_ = Image.open(input_)

# option #3
input_ = "path/to/img.png"

# Initialization
skinsegmenter = SkinDetection()

# Inference
res = skinsegmenter.segment(input_)


# Result
for index, mask in enumerate(res["instance_masks"]):
    cv2.imwrite(f"instance_mask__{index}.png", mask)

cv2.imwrite("integrated_mask.png", res["integrated_mask"])
cv2.imwrite("visualized_image.png", res["visualized_image"])
```

