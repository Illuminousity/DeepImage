import numpy as np
import cv2
import matplotlib.pyplot as plt

def apply_speckle_scrambling(image, speckle_size=10):
    """
    Simulates speckle-like scrambling of an image, similar to multimode fiber output.

    Parameters:
        image (numpy array): Input grayscale image.
        speckle_size (int): Controls the granularity of the speckle pattern.
    
    Returns:
        Scrambled image (numpy array)
    """
    h, w = image.shape

    # Generate a random speckle pattern
    speckle = np.random.randn(h // speckle_size, w // speckle_size)

    # Resize to match the image dimensions
    speckle = cv2.resize(speckle, (w, h), interpolation=cv2.INTER_NEAREST)

    # Multiply the image by the speckle pattern
    scrambled_image = image * speckle

    # Normalize for visibility
    scrambled_image = cv2.normalize(scrambled_image, None, 0, 255, cv2.NORM_MINMAX)

    return scrambled_image.astype(np.uint8)

# Load an example grayscale image
image = cv2.imread("./Programming/DeepImage/Images/Raw/Untitled.png", cv2.IMREAD_GRAYSCALE)

# Apply speckle scrambling
scrambled_image = apply_speckle_scrambling(image, speckle_size=10)

# Display the original and scrambled images
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(image, cmap='gray')
ax[0].set_title("Original Image")
ax[0].axis('off')

ax[1].imshow(scrambled_image, cmap='gray')
ax[1].set_title("Speckle Scrambled Image")
ax[1].axis('off')

plt.show()
