import numpy as np
import cv2
import matplotlib.pyplot as plt
from load_emnist import LoadDataset

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image
import concurrent.futures



def FormatImage(image):

    return cv2.resize(image, (1024, 768), interpolation=cv2.INTER_LINEAR) * (2**8-1)


def apply_gaussian_blur(image, kernel_size=(15,15), sigma=1000):
    """
    Apply Gaussian blur to an image.

    Parameters:
        image: Grayscale NumPy array
        kernel_size: Size of the Gaussian kernel (odd numbers like (5,5), (7,7))
        sigma: Standard deviation of the Gaussian distribution

    Returns:
        Blurred image as a NumPy array
    """
    for i in range(1000):
        image = cv2.GaussianBlur(image, kernel_size, sigma)

    return image

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
emnist = LoadDataset()
#image,label = emnist[1]
#formattedimage = FormatImage(image.numpy().squeeze())
#diffused_image = apply_gaussian_blur(formattedimage)
#fig, ax = plt.subplots(1, 2, figsize=(10, 5))
#ax[0].imshow(formattedimage, cmap='gray')
#ax[0].set_title("Original Image")
#ax[0].axis('off')

#ax[1].imshow(diffused_image, cmap='gray')
#ax[1].set_title("Multimode Fiber Scrambled Image")
#ax[1].axis('off')
#plt.show()


# Function to process a single image
def process_image(i):
    image, label = emnist[i]
    formattedimage = FormatImage(image.numpy().squeeze())
    diffused_image = apply_gaussian_blur(formattedimage)
    
    save_path_diffused = f"./Images/Diffused/Very Diffused/diffused_image{i}.png"
    save_path_raw = f"./Images/Raw/raw_image{i}.png"
    
    cv2.imwrite(save_path_diffused, diffused_image)  # Save diffused image
    cv2.imwrite(save_path_raw, formattedimage)  # Save raw image

    print(f"Image {i} Complete!")

# Use ThreadPoolExecutor for parallel execution
num_workers = 32  # Adjust based on your CPU (recommended: # of CPU cores)
with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
    executor.map(process_image, range(10000))




# Load image in grayscale


    # Apply anisotropic diffusion


    # Open a dialog to select a save directory






