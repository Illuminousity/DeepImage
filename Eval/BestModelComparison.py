import lpips
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torchvision import transforms
import random
import sys

sys.path.append("./NNArchitectures/")
from HybridResNetUNetModel import ResNetUNet
from HybridEfficientNetREDNetModel import EfficientNetREDNet
from HybridResNetREDNetModel import ResNetREDNet
from HybridEfficientNetUNetModel import EfficientNetUNet
from DiffusionDataset import DiffusionDataset
from TestModel import ssim, compute_lpips 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

architectures = {
    "resnet_unet": ResNetUNet,
    "resnet_rednet": ResNetREDNet,
    "effnet_unet": EfficientNetUNet,
    "effnet_rednet": EfficientNetREDNet,
}

best_models = {
    "binary": {
        "120": "effnet_rednet",
        "220": "resnet_rednet",
        "600": "resnet_unet",
        "1500": "effnet_unet",
    },
    "greyscale": {
        "120": "effnet_rednet",
        "220": "effnet_rednet",
        "600": "effnet_rednet",
        "1500": "effnet_unet",
    }
}

grits = ["120", "220", "600", "1500"]
img_types = ["binary", "greyscale"]
transform = transforms.ToTensor()

# Create a compact layout: smaller figure, tighter spacing
fig = plt.figure(figsize=(20, 10)) 
gs = gridspec.GridSpec(4, 5, width_ratios=[1, 1, 1, 1, 0.6], height_ratios=[1, 1, 1, 1],
                       wspace=0.5, hspace=0.8)


test_index = random.randint(0, 999)
ground_truth_imgs = {}
lpips_alex = lpips.LPIPS(net="alex").to(device)

for col, grit in enumerate(grits):
    for row, img_type in enumerate(img_types):
        input_dir = f"./DMD/Testing/{grit} GRIT" if img_type == "binary" else f"./DMD/Greyscale/Testing/{grit} GRIT"
        test_dir= f"./DMD/Testing/RAW" if img_type == "binary" else f"./DMD/Greyscale/Testing/RAW"
        test_dataset = DiffusionDataset(input_dir, test_dir, cap=1000, transform=transform)
        test_input, test_target = test_dataset[test_index]
        test_input = test_input.unsqueeze(0).to(device)
        test_target = test_target.unsqueeze(0).to(device)

        if img_type not in ground_truth_imgs:
            ground_truth_imgs[img_type] = test_target.squeeze().cpu().numpy()

        model_name = best_models[img_type][grit]
        model_class = architectures[model_name]
        model = model_class().to(device)
        model_path = f"./{model_name}_{grit}_60000_l1_{img_type}.pth" if img_type == "greyscale" else f"./{model_name}_{grit}_60000_l1.pth"

        try:
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            model.eval()
            with torch.no_grad():
                output = model(test_input)
        except FileNotFoundError:
            print(f"Missing checkpoint for {model_path}")
            output = test_input.clone() * 0
        threshold = 0.1
        mask = test_target >= threshold
        mask_np = mask.detach().cpu().numpy()  # make sure to detach from autograd if needed
        mask_diffused = output >= threshold
        mask_diffused_np = mask_diffused.detach().cpu().numpy()
        combined_mask = np.logical_or(mask_np,mask_diffused_np)

        # Plot Input
        ax_input = fig.add_subplot(gs[row * 2, col])
        ax_input.imshow(test_input.squeeze().cpu().numpy(), cmap="gray")
        ax_input.axis("off")
        ax_input.set_title(f"{img_type.title()} Input", fontsize=7, pad=2)

        # Plot Output
        ax_output = fig.add_subplot(gs[row * 2 + 1, col])
        ax_output.imshow(output.squeeze().cpu().numpy(), cmap="gray")
        ax_output.axis("off")
        ax_output.set_title(f"Reconstructed:\n{model_name.replace('_', ' ').title()}", fontsize=6, pad=2)
        # Metrics overlay

        ax_output.text(0.98, 0.04,
                    f"SSIM: {ssim(output,test_target,combined_mask):.2f}\nLPIPS: {compute_lpips(lpips_alex,output,test_target):.2f}",
                    fontsize=5, ha='right', va='bottom',
                    color='white', backgroundcolor='black',
                    transform=ax_output.transAxes)


# Ground truth images on the far right
for row, img_type in enumerate(img_types):
    ax_gt = fig.add_subplot(gs[row * 2:(row + 1) * 2, 4])
    ax_gt.imshow(ground_truth_imgs[img_type], cmap="gray")
    ax_gt.set_title(f"{img_type.title()} GT", fontsize=9)
    ax_gt.axis("off")
    

# Add column titles
for i, grit in enumerate(grits):
    ax_title = fig.add_subplot(gs[0, i])
    ax_title.set_title(f"{grit} GRIT", fontsize=9,pad=15)
    ax_title.axis("off")
    

plt.tight_layout()
plt.show()
