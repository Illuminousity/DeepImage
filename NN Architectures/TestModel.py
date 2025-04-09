import torch
import math
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import sys
from HybridResNetUNetModel import ResNetUNet
from HybridEfficientNetREDNetModel import EfficientNetREDNet
from HybridResNetREDNetModel import ResNetREDNet
from HybridEfficientNetUNetModel import EfficientNetUNet
from DiffusionDataset import DiffusionDataset
import matplotlib.pyplot as plt
import numpy as np

# Make sure you've installed LPIPS first: pip install lpips
import lpips

###############################
# SSIM Utility Functions
###############################
def gaussian(window_size, sigma):
    gauss = torch.Tensor([
        math.exp(-(x - window_size//2)**2 / (2 * sigma**2))
        for x in range(window_size)
    ])
    return gauss / gauss.sum()

def create_window(window_size, channel, sigma):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    window = _2D_window.float().unsqueeze(0).unsqueeze(0)
    window = window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, sigma=1.5, val_range=1):
    # Expects img1, img2: (N, C, H, W) in [0, 1]
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel, sigma).to(img1.device)
    
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12   = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2
    
    # Stability constants
    C1 = (0.01 * val_range) ** 2
    C2 = (0.03 * val_range) ** 2

    ssim_map = ((2.0 * mu1_mu2 + C1) * (2.0 * sigma12 + C2)) / (
                (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
               )
    
    return ssim_map.mean()

###############################
# LPIPS for Grayscale
###############################
def compute_lpips(model_lpips, img_pred, img_gt):
    """
    Compute LPIPS using the specified model (e.g., lpips.LPIPS(net='alex')).
    Expects each image in (N, C=1, H, W) range [0,1].
    LPIPS default expects RGB in range [-1,1], so we'll do:
      1) replicate channel 3 times (for grayscale -> RGB)
      2) scale from [0,1] to [-1,1]

    Returns average LPIPS for the batch (scalar).
    """
    # 1) replicate the single channel into 3 channels
    img_pred_3ch = img_pred.repeat(1, 3, 1, 1)
    img_gt_3ch   = img_gt.repeat(1, 3, 1, 1)
    
    # 2) shift from [0,1] -> [-1,1]
    img_pred_3ch = (img_pred_3ch * 2.0) - 1.0
    img_gt_3ch   = (img_gt_3ch * 2.0) - 1.0

    lpips_val = model_lpips(img_pred_3ch, img_gt_3ch)
    # lpips() returns (N,1,1,1). We'll take the mean for the batch
    return lpips_val.mean().item()

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python validate_model.py <GRIT_VALUE> <MODEL_PATH>")
        sys.exit(1)
    
    grit_value = sys.argv[1]
    model_path = sys.argv[2]
    diffused_dir = f"./DMD/Greyscale/Testing/{grit_value} GRIT"
    clean_dir = "./DMD/Greyscale/Testing/Raw"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the model (adjust model type as needed)
    model = EfficientNetREDNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Initialize LPIPS with a pre-trained "alex" backbone
    lpips_alex = lpips.LPIPS(net='alex').to(device)

    # Transforms
    transform = transforms.Compose([
        transforms.ToTensor()  # gives us [0,1] grayscale images
    ])
    
    # Create dataset & loader
    dataset = DiffusionDataset(diffused_dir, clean_dir, cap=1000, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    total_ssim = 0.0
    total_lpips = 0.0
    
    # Evaluate the model
    with torch.no_grad():
        for i, (diffused, clean) in enumerate(dataloader):
            diffused, clean = diffused.to(device), clean.to(device)
            output = model(diffused)

            # Compute SSIM
            current_ssim = ssim(output, clean)
            total_ssim += current_ssim.item()

            # Compute LPIPS
            current_lpips = compute_lpips(lpips_alex, output, clean)
            total_lpips += current_lpips

            print(f"Batch {i}: SSIM = {current_ssim:.6f}, LPIPS = {current_lpips:.6f}")
            
            # Display a sample prediction (optional)
            if i == 786:
                output_np = output.squeeze().cpu().numpy()
                clean_np = clean.squeeze().cpu().numpy()
                diffused_np = diffused.squeeze().cpu().numpy()

                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                axs[0].imshow(diffused_np, cmap='gray')
                axs[0].set_title("Diffused Input")
                axs[1].imshow(output_np, cmap='gray')
                axs[1].set_title("Net Output")
                axs[2].imshow(clean_np, cmap='gray')
                axs[2].set_title("Ground Truth")
                
                for ax in axs:
                    ax.axis('off')
                plt.show()
    
    avg_ssim = total_ssim / len(dataloader)
    avg_lpips = total_lpips / len(dataloader)

    print(f"Validation Completed.\n"
          f"Average SSIM:  {avg_ssim:.6f}\n"
          f"Average LPIPS: {avg_lpips:.6f}  (lower = better)")

