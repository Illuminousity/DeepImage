import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Data Collection/')))

from load_emnist import LoadDataset, GetImage

# Load the EMNIST dataset
emnist = LoadDataset(mode=False)  # mode=False means test set

# Pick an image (e.g., the first one)
img = GetImage(1000, emnist).astype(np.float32)
img_8bit = (img * 255).astype(np.uint8)

# Apply a vertical gradient
gradient = np.linspace(1, 0, img_8bit.shape[0]).reshape(-1, 1)
gradient = np.tile(gradient, (1, img_8bit.shape[1]))
img_gradient = (img_8bit * gradient).astype(np.uint8)

# Function to split into bitplanes
def split_into_bitplanes(image):
    bitplanes = []
    for bit in range(8):
        plane = (image >> bit) & 1
        bitplanes.append(plane)
    return bitplanes

# Function to recombine bitplanes
def recombine_bitplanes(bitplanes):
    recombined = np.zeros_like(bitplanes[0], dtype=np.uint8)
    for bit, plane in enumerate(bitplanes):
        recombined += (plane.astype(np.uint8) << bit)
    return recombined

# Split the gradient image into bitplanes
bitplanes = split_into_bitplanes(img_gradient)

# Plot original, bitplanes, and recombined image
fig, axes = plt.subplots(3, 3, figsize=(12, 12))

# Original image with gradient
axes[0, 0].imshow(img_gradient, cmap='gray')
axes[0, 0].set_title('Gradient Image')
axes[0, 0].axis('off')

# Plot bitplanes (MSB first)
for i in range(8):
    row = (i+1)//3
    col = (i+1)%3
    axes[row, col].imshow(bitplanes[7-i], cmap='gray')  # show MSB first
    axes[row, col].set_title(f'Bitplane {7-i}')
    axes[row, col].axis('off')

# Recombined image
reconstructed = recombine_bitplanes(bitplanes)
axes[2, 2].imshow(reconstructed, cmap='gray')
axes[2, 2].set_title('Recombined Image')
axes[2, 2].axis('off')

plt.tight_layout()
plt.show()
