import torch
import matplotlib.pyplot as plt
import numpy as np
import random
from torchvision import transforms
import sys

sys.path.append("./NNArchitectures/")
from DiffusionDataset import DiffusionDataset

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GRIT levels you want to compare
grit_levels = ["120", "220", "600", "1500"]

# Randomly select one index (same across all datasets)
transform = transforms.ToTensor()
sample_index = None

diffused_images = []
ground_truth = None

for grit in grit_levels:
    dataset = DiffusionDataset(f"./DMD/Greyscale/Testing/{grit} GRIT", "./DMD/Greyscale/Testing/Raw", cap=1000, transform=transform)
    if sample_index is None:
        sample_index = random.randint(0, len(dataset) - 1)
    diffused_img, gt = dataset[sample_index]
    diffused_images.append(diffused_img)
    ground_truth = gt  # same for all, just overwrite

# === Plotting ===
fig = plt.figure(figsize=(10, 6))
gs = fig.add_gridspec(2, 3, width_ratios=[1.2, 1, 1], wspace=0.3, hspace=0.3)

# Ground truth on the left
ax_gt = fig.add_subplot(gs[:, 0])
ax_gt.imshow(ground_truth.squeeze().numpy(), cmap='gray')
ax_gt.set_title("Ground Truth", fontsize=14)
ax_gt.axis('off')

# 2x2 grid for grit diffusions
for i, (grit, img) in enumerate(zip(grit_levels, diffused_images)):
    row = i // 2
    col = i % 2
    ax = fig.add_subplot(gs[row, col + 1])
    ax.imshow(img.squeeze().numpy(), cmap='gray')
    ax.set_title(f"{grit} GRIT", fontsize=12)
    ax.axis('off')

plt.suptitle("Scattering of Light by GRIT level", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("grit_diffusion_comparison.png", dpi=300)
plt.show()
