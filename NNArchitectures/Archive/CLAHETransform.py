import cv2
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

class CLAHETransform:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, img):
        if isinstance(img, Image.Image):
            img = np.array(img)
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        clahe_img = self.clahe.apply(img)
        return TF.to_tensor(Image.fromarray(clahe_img))
