import numpy as np
import cv2
import matplotlib.pyplot as plt

import numpy as np
import cv2
import matplotlib.pyplot as plt
from image_to_array import LoadImage
from load_emnist import GetImage

def multimode_fiber_scramble(image, num_modes=50, speckle_size=10):
    """
    Simulates multimode fiber scrambling using:
      1. Mode-specific Gaussian phase distribution
      2. Random transmission matrix (T)
      3. Speckle-like output pattern

    Parameters:
        image (numpy array): Input grayscale image.
        num_modes (int): Number of spatial modes (higher = stronger scrambling).
        speckle_size (int): Controls the granularity of the speckle pattern.
    
    Returns:
        Scrambled image (numpy array).
    """
    image = image.astype(np.float32)
    h, w = image.shape

    # Step 1: Generate a Fiber-Specific Mode Distribution (Gaussian in Fourier space)
    X, Y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    radius = np.sqrt(X**2 + Y**2)
    gaussian_mask = np.exp(-radius**2 * (num_modes / 10.0))  # Adjust for mode complexity

    # Step 2: Generate a Random Phase Mask for Mode Mixing
    random_phase = np.exp(1j * np.random.uniform(0, 2 * np.pi, (h, w)))
    
    # Step 3: Generate a Random Transmission Matrix (T)
    T_real = np.random.randn(h, w) * gaussian_mask  # Apply Gaussian-weighted randomness
    T_imag = np.random.randn(h, w) * gaussian_mask
    T = T_real + 1j * T_imag  # Complex transmission matrix

    # Step 4: Apply Fourier Transform to Move to the Frequency Domain
    fft_image = np.fft.fft2(image)

    # Step 5: Apply Transmission Matrix (T) and Phase Mask
    scrambled_fft = fft_image * T * random_phase

    # Step 6: Apply Inverse Fourier Transform to Return to Spatial Domain
    scrambled_image = np.abs(np.fft.ifft2(scrambled_fft))

    # Step 7: Apply a Speckle-Like Intensity Pattern
    speckle = np.random.randn(h // speckle_size, w // speckle_size)
    speckle = cv2.resize(speckle, (w, h), interpolation=cv2.INTER_NEAREST)
    scrambled_image *= speckle

    # Step 8: Normalize to 0-255 for visualization
    scrambled_image = cv2.normalize(scrambled_image, None, 0, 255, cv2.NORM_MINMAX)

    return scrambled_image.astype(np.uint8)


# Load a grayscale image
image = GetImage(6)

# Apply multimode fiber scrambling
scrambled_image = multimode_fiber_scramble(image, num_modes=10, speckle_size=1)

# Display results
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(image, cmap='gray')
ax[0].set_title("Original Image")
ax[0].axis('off')

ax[1].imshow(scrambled_image, cmap='gray')
ax[1].set_title("Multimode Fiber Scrambled Image")
ax[1].axis('off')

plt.show()
