import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color, filters, util
from skimage.transform import resize
from scipy.ndimage import gaussian_filter
from load_emnist import GetImage

# Load an example image and convert to grayscale
image = GetImage(15)

# Apply Gaussian Blur (Light Diffusion)
diffused_image = gaussian_filter(image, sigma=2.5)  # Higher sigma = more diffusion



# Display images
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].imshow(image, cmap='gray')
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(diffused_image, cmap='gray')
axes[1].set_title("Diffused Image (Gaussian Blur)")
axes[1].axis("off")


plt.show()
