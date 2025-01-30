import numpy as np
import cv2
import matplotlib.pyplot as plt
from image_to_array import convert_single

def apply_mode_scrambling(image, num_modes=50):
    """
    Simulates multimode fiber scrambling using a random phase mask.
    
    Parameters:
        image (numpy array): Input grayscale image.
        num_modes (int): Number of spatial modes to simulate (higher = more scrambling).
    
    Returns:
        Scrambled image (numpy array)
    """
    # Convert to float32 for Fourier Transform
    image = image.astype(np.float32)

    # Get image dimensions
    h, w = image.shape

    # Generate a complex random phase mask
    phase_mask = np.exp(1j * np.random.uniform(0, 2 * np.pi, (h, w)))

    # Apply Fourier Transform to get spatial frequency domain
    fft_image = np.fft.fft2(image)

    # Simulate mode mixing by applying random phase distortions
    scrambled_fft = fft_image * phase_mask

    # Apply Inverse Fourier Transform to return to spatial domain
    scrambled_image = np.abs(np.fft.ifft2(scrambled_fft))

    # Normalize to 0-255
    scrambled_image = cv2.normalize(scrambled_image, None, 0, 255, cv2.NORM_MINMAX)

    return scrambled_image.astype(np.uint8)

# Load an example grayscale image
image = cv2.imread("./Programming/DeepImage/Images/Raw/Untitled.png", cv2.IMREAD_GRAYSCALE)

# Apply mode scrambling
scrambled_image = apply_mode_scrambling(image, num_modes=50)

# Display the original and scrambled images
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(image, cmap='gray')
ax[0].set_title("Original Image")
ax[0].axis('off')

ax[1].imshow(scrambled_image, cmap='gray')
ax[1].set_title("Mode Scrambled Image")
ax[1].axis('off')

plt.show()
