import numpy as np
import matplotlib.pyplot as plt
import cv2

# Mock functions to simulate EMNIST loading
# Replace with your actual GetImage and LoadDataset imports
from torchvision.datasets import EMNIST
from torchvision import transforms
import torch

# Load a small EMNIST dataset (test set)
transform = transforms.Compose([
    transforms.ToTensor()
])
emnist_data = EMNIST(root='./data', split='letters', download=True, train=False, transform=transform)

def get_emnist_image(index):
    img, label = emnist_data[index]
    img = img.squeeze(0).numpy()  # Remove channel dimension, to (28, 28)
    return img

# Function to create gradient
def create_gradient(height, width):
    gradient = np.linspace(1, 0, height).reshape(-1, 1)
    gradient = np.tile(gradient, (1, width))
    return gradient

# Main plotting function
def plot_emnist_gradient(index):
    # Load EMNIST image
    emnist_img = get_emnist_image(index)
    emnist_img_resized = cv2.resize(emnist_img, (256, 192), interpolation=cv2.INTER_LINEAR)

    # Generate gradient
    gradient = create_gradient(192, 256)

    # Apply gradient to the resized EMNIST image
    
    emnist_with_gradient = emnist_img_resized * gradient

    # Plotting
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].imshow(emnist_img_resized, cmap='gray')
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    axs[1].imshow(gradient, cmap='gray')
    axs[1].set_title('Greyscale Gradient Filter')
    axs[1].axis('off')

    axs[2].imshow(emnist_with_gradient, cmap='gray')
    axs[2].set_title('Image with Gradient')
    axs[2].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_emnist_gradient(index=184)  # Change index if you want different images
