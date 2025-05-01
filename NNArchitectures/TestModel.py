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
from scipy.ndimage import binary_dilation
# Make sure you've installed LPIPS first: pip install lpips
import lpips
import argparse

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

def ssim(img1, img2, mask, window_size=11, sigma=1.5, val_range=1):
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
  
   
    return ssim_map[mask].mean() 

###############################
# LPIPS
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
    

    parser = argparse.ArgumentParser(description="Run CNN image reconstruction with specified GRIT and model path.")
    parser.add_argument('--grit_value', type=int, required=True, help='GRIT value for the diffuser (e.g., 1500)')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model .pth file')
    parser.add_argument("--greyscale", action="store_true")
    parser.add_argument("--architecture", type=str, default="effnet_unet")

    args = parser.parse_args()
    architecture = args.architecture
    greyscale = args.greyscale
    grit_value = args.grit_value
    model_path = args.model_path
    if greyscale:
        diffused_dir = f"./DMD/Greyscale/Testing/{grit_value} GRIT"
        clean_dir = "./DMD/Greyscale/Testing/Raw"
    else:
        diffused_dir = f"./DMD/Testing/{grit_value} GRIT"
        clean_dir = "./DMD/Testing/Raw"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the model (adjust model type as needed)
    match architecture:
        case "effnet_unet": 
            model = EfficientNetUNet().to(device)
        case "resnet_unet": 
            model = ResNetUNet().to(device)
        case "effnet_rednet": 
            model = EfficientNetREDNet().to(device)
        case "resnet_rednet": 
            model = ResNetREDNet().to(device)
        case _: 
            print("Invalid architecture option")
            exit()
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
    threshold = 0.1
    # Evaluate the model
    with torch.no_grad():
        for i, (diffused, clean) in enumerate(dataloader):
            diffused, clean = diffused.to(device), clean.to(device)
            output = model(diffused)

            # Compute SSIM
            mask = clean >= threshold
            mask_np = mask.detach().cpu().numpy()  # make sure to detach from autograd if needed
            mask_diffused = output >=threshold
            mask_diffused_np = mask_diffused.detach().cpu().numpy()
            combined_mask = np.logical_or(mask_np,mask_diffused_np)
            current_ssim = ssim(output, clean,mask)
            total_ssim += current_ssim.item()

            # Compute LPIPS
            current_lpips = compute_lpips(lpips_alex, output, clean)
            total_lpips += current_lpips

            print(f"Batch {i}: SSIM = {current_ssim:.6f}, LPIPS = {current_lpips:.6f}")
            

    
    avg_ssim = total_ssim / len(dataloader)
    avg_lpips = total_lpips / len(dataloader)

    print(f"Validation Completed.\n"
          f"Average SSIM:  {avg_ssim:.6f}\n"
          f"Average LPIPS: {avg_lpips:.6f}  (lower = better)")

