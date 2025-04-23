import torch
import matplotlib.pyplot as plt
import random
import lpips
import matplotlib.gridspec as gridspec
from torchvision import transforms
import sys

sys.path.append("./NNArchitectures/")
from HybridResNetUNetModel import ResNetUNet
from HybridEfficientNetREDNetModel import EfficientNetREDNet
from HybridResNetREDNetModel import ResNetREDNet
from HybridEfficientNetUNetModel import EfficientNetUNet
from DiffusionDataset import DiffusionDataset
from TestModel import ssim, compute_lpips 

"""
Visualise **binary**‑image reconstructions at **600 GRIT** for models trained on
**60 000 images** with either **L1** or **NPCC** loss.

Grid layout:
    rows  = architectures × loss‑types  (e.g. ResNet‑UNet [L1] just above ResNet‑UNet [NPCC])
    cols  = single 60 k checkpoint
    first column + last column show the diffused input and ground‑truth.

File‑name convention assumed:
    ./<arch>_<grit>_60000_<loss>_binary.pth
Example:
    ./resnet_unet_600_60000_npcc_binary.pth
Adjust `model_path` formatting if your checkpoints follow a different scheme.
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

architectures = {
    "resnet_unet": ResNetUNet,
    "resnet_rednet": ResNetREDNet,
    "effnet_unet": EfficientNetUNet,
    "effnet_rednet": EfficientNetREDNet,
}

loss_types = ["l1", "npcc"]

dataset_sizes = [60000]  # single checkpoint only
grit_value = "600"  # change here for other diffusers

# === Prepare a single random binary test pattern ===
transform = transforms.ToTensor()

test_dataset = DiffusionDataset(
    f"./DMD/Testing/{grit_value} GRIT",
    "./DMD/Testing/Raw",
    cap=1000,
    transform=transform,
)
index = random.randint(0, len(test_dataset) - 1)
test_input, test_target = test_dataset[index]

test_input = test_input.unsqueeze(0).to(device)
test_target = test_target.unsqueeze(0).to(device)

lpips_alex = lpips.LPIPS(net="alex").to(device)

# === Figure layout ===
n_rows = 3
n_cols = len(loss_types)

fig = plt.figure(figsize=(n_cols * 3, n_rows * 3))
side_ratio   = 1          # 0.8 – 1.2 is typical
middle_ratio = 1        # start big, dial back later

width_ratios = [side_ratio] + [middle_ratio] * n_cols + [side_ratio]
gs = gridspec.GridSpec(
    n_rows,
    n_cols+2,
    width_ratios=width_ratios,
    wspace=0.1,
    hspace=0.4
)

# Shared diffused input
ax_input = fig.add_subplot(gs[:, 0])
ax_input.imshow(test_input.squeeze().cpu().numpy(), cmap="gray")
ax_input.axis("off")
ax_input.set_title(f"{grit_value} GRIT input", fontsize=14)

# Shared ground truth
ax_gt = fig.add_subplot(gs[:, -1])
ax_gt.imshow(test_target.squeeze().cpu().numpy(), cmap="gray")
ax_gt.axis("off")
ax_gt.set_title("Ground truth", fontsize=14)

# === Model outputs ===

for loss_idx, loss_tag in enumerate(loss_types):
        ax = fig.add_subplot(gs[:, loss_idx+1])

        for col_idx, size in enumerate(dataset_sizes):
            # ---- Axes placement
           

            # ---- Load model
            model_path = f"./resnet_rednet_{grit_value}_{size}_{loss_tag}.pth"
            model_class = ResNetREDNet
            model = model_class().to(device)
            try:
                state_dict = torch.load(model_path, map_location=device)
            except FileNotFoundError:
                ax.text(0.5, 0.5, "ckpt\nmissing", ha="center", va="center", fontsize=8)
                ax.axis("off")
                continue

            model.load_state_dict(state_dict)
            model.eval()

            # ---- Inference & metrics
            with torch.no_grad():
                output = model(test_input)
                mask = (test_target >= 0.1) | (output >= 0.1)
                ssim_val = ssim(output, test_target, mask).item()
                lpips_val = compute_lpips(lpips_alex, output, test_target)

            # ---- Draw
            ax.imshow(output.squeeze().cpu().numpy(), cmap="gray")
            ax.axis("off")

            ax.set_title(f"{loss_tag.upper()}", fontsize=11)

            ax.text(
                1,
                0,
                f"{loss_tag.upper()}\nSSIM {ssim_val:.2f}\nLPIPS {lpips_val:.2f}",
                fontsize=6,
                ha="right",
                va="bottom",
                color="white",
                backgroundcolor="black",
                transform=ax.transAxes,
            )


plt.savefig("binary_600Grit_60k_L1_vs_NPCC.png", dpi=300)
plt.show()
