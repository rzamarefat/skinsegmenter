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

![skin drawio](https://github.com/rzamarefat/skinsegmenter/assets/79300456/9374f9a7-c66d-4b2c-9317-4c7ef01d5e07)

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
input_ = cv2.imread(input_)

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

