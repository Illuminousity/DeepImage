import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.ndimage import convolve
import numpy as np

# Set a seed for reproducibility (optional)


def draw_convolution_step(kernel, stride=1):
    np.random.seed(42)

    # Generate a 10x10 binary matrix (values 0 or 1)
    input_matrix = np.random.randint(0, 2, size=(10, 10))
    input_matrix = np.array(input_matrix)
    kernel = np.array(kernel)
    kernel_size = kernel.shape[0]



    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    # Input
    ax[0].imshow(input_matrix, cmap='Blues', vmin=0, vmax=1.5)
    ax[0].set_title('Input Image')
    for i in range(input_matrix.shape[0]):
        for j in range(input_matrix.shape[1]):
            ax[0].text(j, i, str(input_matrix[i, j]), ha='center', va='center')

    # Kernel
    ax[1].imshow(kernel, cmap='Reds', vmin=0, vmax=1.5)
    ax[1].set_title('Convolution Kernel')
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            ax[1].text(j, i, str(kernel[i, j]), ha='center', va='center')

    # Convolution Operation
    output = convolve(input_matrix,kernel)

    ax[2].imshow(output, cmap='Greens')
    ax[2].set_title('Output Feature Map')
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            ax[2].text(j, i, f"{output[i, j]:.1f}", ha='center', va='center')

    for a in ax:
        a.set_xticks(np.arange(-0.5, 4, 1), minor=True)
        a.set_yticks(np.arange(-0.5, 4, 1), minor=True)
        a.grid(which="minor", color="black", linestyle='-', linewidth=0.5)
        a.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    plt.tight_layout()
    plt.show()


# Example


kernel = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
]

draw_convolution_step(kernel)
