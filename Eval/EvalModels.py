import os
import re
import csv
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import lpips
import numpy as np
import sys

sys.path.append("./NNArchitectures/")
from HybridResNetUNetModel import ResNetUNet
from HybridEfficientNetREDNetModel import EfficientNetREDNet
from HybridResNetREDNetModel import ResNetREDNet
from HybridEfficientNetUNetModel import EfficientNetUNet
from DiffusionDataset import DiffusionDataset
from TestModel import ssim, compute_lpips 



###############################
# SSIM Utility Functions
###############################
def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2 / (2 * sigma**2))
                            for x in range(window_size)])
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
# LPIPS Function (NN score)
###############################
def compute_lpips(model_lpips, img_pred, img_gt):
    # Replicate grayscale to 3 channels and scale images from [0,1] to [-1,1]
    img_pred_3ch = img_pred.repeat(1, 3, 1, 1)
    img_gt_3ch   = img_gt.repeat(1, 3, 1, 1)
    img_pred_3ch = (img_pred_3ch * 2.0) - 1.0
    img_gt_3ch   = (img_gt_3ch * 2.0) - 1.0

    lpips_val = model_lpips(img_pred_3ch, img_gt_3ch)
    return lpips_val.mean().item()

###############################
# Model Mapping Dictionary
###############################
model_map = {
    "resnet_unet": ResNetUNet,
    "resnet_rednet": ResNetREDNet,
    "effnet_unet": EfficientNetUNet,
    "effnet_rednet": EfficientNetREDNet
}

###############################
# Evaluation Routine
###############################
def evaluate_model(grit_value, greyscale, model_path, lpips_alex, device):
    # Define the dataset directories

    threshold = 0.1

    if greyscale:
        diffused_dir = f"./DMD/Greyscale/Testing/{grit_value} GRIT"
        clean_dir = "./DMD/Greyscale/Testing/Raw"
    else:
        diffused_dir = f"./DMD/Testing/{grit_value} GRIT"
        clean_dir = "./DMD/Testing/Raw"
    
    # Parse the filename to get the architecture.
    basename = os.path.basename(model_path)
    # Expected pattern: <architecture>_<GRIT>_<datasetSize>_...
    match = re.match(r"(.+?)_(\d+)_(\d+).+\.pth", basename)
    if not match:
        print(f"Could not parse filename: {basename}")
        return None, None, None
    arch_str = match.group(1)
    
    if arch_str not in model_map:
        print(f"Unrecognized architecture {arch_str} in {basename}")
        return None, None, None
    
    # Instantiate and load the model.
    model_cls = model_map[arch_str]
    model = model_cls().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    

    
    # Data transformations
    transform = transforms.Compose([transforms.ToTensor()])
    
    # Create the dataset and loader
    dataset = DiffusionDataset(diffused_dir, clean_dir, cap=1000, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    

    

    total_ssim = 0.0
    total_lpips = 0.0
    with torch.no_grad():
        for diffused, clean in dataloader:
            diffused, clean = diffused.to(device), clean.to(device)
            output = model(diffused)
            mask = clean >= threshold
            mask_np = mask.detach().cpu().numpy()  # make sure to detach from autograd if needed
            mask_diffused = output >=threshold
            mask_diffused_np = mask_diffused.detach().cpu().numpy()
            combined_mask = np.logical_or(mask_np,mask_diffused_np)
            current_ssim = ssim(output, clean,combined_mask)
            total_ssim += current_ssim.item()
            current_lpips = compute_lpips(lpips_alex, output, clean)
            total_lpips += current_lpips
            
    avg_ssim = total_ssim / len(dataloader)
    avg_lpips = total_lpips / len(dataloader)
    
    return arch_str, avg_ssim, avg_lpips

###############################
# Main Loop: Evaluate all models and log results to CSV
###############################
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Define where your model files are stored.
    model_dir = "./"  # Adjust this directory as needed.
    # Create the output folder for CSV evaluation logs.
    csv_folder = "./csveval"
    os.makedirs(csv_folder, exist_ok=True)
    csv_path = os.path.join(csv_folder, "evaluation_results_new.csv")
    # Setup LPIPS model (using the pre-trained AlexNet backbone)
    lpips_alex = lpips.LPIPS(net='alex').to(device)
    # Open CSV file for writing.
    with open(csv_path, mode="w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write header: ModelName, Architecture, GRIT, ModelPath, Avg SSIM, Avg LPIPS
        csv_writer.writerow(["ModelName", "Architecture", "GRIT", "ModelPath", "Average_SSIM", "Average_LPIPS"])
        
        # Loop over each model file in the directory 
        for root, _, files in os.walk(model_dir):
            for file in files:
                if file.endswith(".pth"):
                    model_path = os.path.join(root, file)
                    # Parse GRIT value from the filename (assumed to be the second group in the filename)
                    match = re.match(r"(.+?)_(\d+)_(\d+).+\.pth", file)
                    if not match:
                        print(f"Skipping unrecognized file: {file}")
                        continue
                    grit_value = match.group(2)
                    greyscale = False

                    if 'greyscale' in file:
                        greyscale = True
                    # Evaluate the model.
                    print(f"Starting to evaluate: {file}")
                    arch, avg_ssim, avg_lpips = evaluate_model(grit_value, greyscale, model_path, lpips_alex, device)
                    if arch is None:
                        continue
                    
                    # Write the results to CSV.
                    print(f"Finished Evaluating: {file}")
                    csv_writer.writerow([file, arch, grit_value, model_path, f"{avg_ssim:.6f}", f"{avg_lpips:.6f}"])
                    print(f"Evaluated {file}: SSIM = {avg_ssim:.6f}, LPIPS = {avg_lpips:.6f}")
    
    print(f"Evaluation CSV saved to: {csv_path}")

if __name__ == '__main__':
    main()
