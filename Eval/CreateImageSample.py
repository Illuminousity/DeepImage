import torch
import matplotlib.pyplot as plt
import numpy as np
import random
from torchvision import transforms
from torch.utils.data import DataLoader
import lpips
import matplotlib.gridspec as gridspec
import sys

sys.path.append("./NNArchitectures/")
from HybridResNetUNetModel import ResNetUNet
from HybridEfficientNetREDNetModel import EfficientNetREDNet
from HybridResNetREDNetModel import ResNetREDNet
from HybridEfficientNetUNetModel import EfficientNetUNet
from DiffusionDataset import DiffusionDataset
from TestModel import ssim, compute_lpips 


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

architectures = {
    "resnet_unet": ResNetUNet,
    "resnet_rednet": ResNetREDNet,
    "effnet_unet": EfficientNetUNet,
    "effnet_rednet": EfficientNetREDNet
}

dataset_sizes = [10000, 20000, 30000, 40000, 50000, 60000]
grit_value = "120"

# === Prepare test image (same for all runs) ===
transform = transforms.ToTensor()
test_dataset = DiffusionDataset(f"./DMD/Greyscale/Testing/{grit_value} GRIT", "./DMD/Greyscale/Testing/Raw",cap=1000, transform=transform)
index= 397
test_input, test_target = test_dataset[index]
test_input = test_input.unsqueeze(0).to(device)
test_target = test_target.unsqueeze(0).to(device)

lpips_alex = lpips.LPIPS(net='alex').to(device)

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

n_rows = len(architectures)
n_cols = len(dataset_sizes)

fig = plt.figure(figsize=(n_cols * 3 + 5, n_rows * 3))
gs = gridspec.GridSpec(n_rows, n_cols + 2, 
                       width_ratios=[1.2] + [1]*n_cols + [1.2],
                       wspace=0.1, hspace=0.3)

# === Shared diffused input ===
ax_input = fig.add_subplot(gs[:, 0])
ax_input.imshow(test_input.squeeze().cpu().numpy(), cmap='gray')
ax_input.axis('off')
ax_input.set_title("120 GRIT Input", fontsize=14)

# === Shared ground truth ===
ax_gt = fig.add_subplot(gs[:, -1])
ax_gt.imshow(test_target.squeeze().cpu().numpy(), cmap='gray')
ax_gt.axis('off')
ax_gt.set_title("Ground Truth", fontsize=14)
threshold = 0.1
# === Model outputs
for row_idx, (arch_name, model_class) in enumerate(architectures.items()):
    for col_idx, size in enumerate(dataset_sizes):
        ax = fig.add_subplot(gs[row_idx, col_idx + 1])
        model_path = f"./{arch_name}_120_{size}_l1_greyscale.pth"
        model = model_class().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        with torch.no_grad():
            output = model(test_input)
            mask = test_target >= threshold
            mask_np = mask.detach().cpu().numpy()  # make sure to detach from autograd if needed
            mask_diffused = output >=threshold
            mask_diffused_np = mask_diffused.detach().cpu().numpy()
            combined_mask = np.logical_or(mask_np,mask_diffused_np)
            ssim_val = ssim(output, test_target, combined_mask).item()
            lpips_val = compute_lpips(lpips_alex, output, test_target)

            ax.imshow(output.squeeze().cpu().numpy(), cmap='gray')
            ax.axis('off')

            # Column title (only for top row)
            if row_idx == 0:
                ax.set_title(f"{size//1000}k Images", fontsize=12)

            # Metrics overlay
            ax.text(0.98, 0.04,
                    f"SSIM: {ssim_val:.2f}\nLPIPS: {lpips_val:.2f}",
                    fontsize=8, ha='right', va='bottom',
                    color='white', backgroundcolor='black',
                    transform=ax.transAxes)

# === Architecture labels outside plot (clean style)
for row_idx, arch_name in enumerate(architectures):
    fig.text(0.1, 1 - (row_idx + 0.5) / n_rows,
             arch_name.replace("_", "\n"),
             va='center', ha='right', fontsize=12, rotation=90)

# === Final layout and save
fig.subplots_adjust(left=0.12, right=0.95, top=0.92, bottom=0.05)
plt.savefig("model_comparison_grid.png", dpi=300)
plt.show()
