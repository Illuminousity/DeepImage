import numpy as np
import cv2
import matplotlib.pyplot as plt
from image_to_array import convert_single


import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image


def apply_gaussian_blur(image, kernel_size=(5, 5), sigma=0.5):
    """
    Apply Gaussian blur to an image.

    Parameters:
        image: Grayscale NumPy array
        kernel_size: Size of the Gaussian kernel (odd numbers like (5,5), (7,7))
        sigma: Standard deviation of the Gaussian distribution

    Returns:
        Blurred image as a NumPy array
    """
    return cv2.GaussianBlur(image, kernel_size, sigma)

def anisotropic_diffusion(image, iterations=1000, kappa=10, gamma=0.2, option=2):
    """
    Perform anisotropic diffusion (Perona-Malik filtering) on an image.
    """
    image = image.astype(np.float32)
    for _ in range(iterations):
        # Compute gradients
        north = np.roll(image, -1, axis=0) - image
        south = np.roll(image, 1, axis=0) - image
        east = np.roll(image, -1, axis=1) - image
        west = np.roll(image, 1, axis=1) - image

        if option == 1:
            c_north = np.exp(-(north / kappa) ** 2)
            c_south = np.exp(-(south / kappa) ** 2)
            c_east = np.exp(-(east / kappa) ** 2)
            c_west = np.exp(-(west / kappa) ** 2)
        else:
            c_north = 1 / (1 + (north / kappa) ** 2)
            c_south = 1 / (1 + (south / kappa) ** 2)
            c_east = 1 / (1 + (east / kappa) ** 2)
            c_west = 1 / (1 + (west / kappa) ** 2)

        image += gamma * (
            c_north * north + c_south * south + c_east * east + c_west * west
        )

    return np.clip(image, 0, 255).astype(np.uint8)

# Open a file dialog to select an image

# Load image in grayscale
image = convert_single()

    # Apply anisotropic diffusion
diffused_image = apply_gaussian_blur(image)

    # Open a dialog to select a save directory



save_path = "./Programming/DeepImage/Images/Diffused/diffused_image.png"
cv2.imwrite(save_path, diffused_image)  # Save the image
print(f"Image saved successfully at: {save_path}")


