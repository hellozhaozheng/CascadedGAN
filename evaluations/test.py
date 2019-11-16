import numpy as np
from scipy.misc import imread

f = "/home/zerozone/Works/Grad_Project/SPADE-master/datasets/coco_stuff/val_img/000000001268.jpg"
image = np.array(imread(str(f)).astype(np.float32))
print(image.shape)
# images = np.array([imread(str(f)).astype(np.float32)
                   # for f in files[start:end]])
