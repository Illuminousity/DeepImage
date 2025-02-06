import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color, io
from skimage.transform import resize
from load_emnist import GetImage

# Seed for reproducable TM's
SEED = 1
np.random.seed(SEED)

# Load an example image and convert to grayscale
image = GetImage(10)

# Convert image to a 1D vector
image_vector = image.flatten()

# Generate a random transmission matrix
size = image_vector.shape[0]
T = np.random.rand(size, size)

# Ensure that T is invertible
while np.linalg.det(T) == 0:
    T = np.random.rand(size, size)
# Scramble the image using the transmission matrix
scrambled_vector = T @ image_vector

# Compute the inverse of the transmission matrix
T_inv = np.linalg.inv(T)

# Recover the original image
recovered_vector = T_inv @ scrambled_vector
recovered_image = recovered_vector.reshape(image.shape)

# Display images
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(image, cmap='gray')
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(scrambled_vector.reshape(image.shape), cmap='gray')
axes[1].set_title("Scrambled Image")
axes[1].axis("off")

axes[2].imshow(recovered_image, cmap='gray')
axes[2].set_title("Recovered Image")
axes[2].axis("off")

plt.show()
