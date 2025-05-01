import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
from torchvision import transforms
from PIL import Image
import sys
import os
# SSIM components
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

def ssim_with_map(img1, img2, window_size=11, sigma=1.5, val_range=1):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel, sigma).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = (0.01 * val_range) ** 2
    C2 = (0.03 * val_range) ** 2

    ssim_map = ((2.0 * mu1_mu2 + C1) * (2.0 * sigma12 + C2)) / (
                (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean(), ssim_map

def load_image(path):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
    img = Image.open(path)
    return transform(img).unsqueeze(0)  # Add batch dimension

if __name__ == "__main__":

    image_path = "./DMD/Testing/120 GRIT/captured_frame_725.png"
    gt_path = "./DMD/Testing/RAW/captured_frame_725.png"
    model_path = "./resnet_rednet_120_60000_l1.pth"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    sys.path.append("./NNArchitectures/")    
    from HybridResNetREDNetModel import ResNetREDNet 
    model = ResNetREDNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load images
    input_img = load_image(image_path).to(device)
    ground_truth = load_image(gt_path).to(device)

    # Inference
    with torch.no_grad():
        output = model(input_img)
        ssim_val, ssim_map = ssim_with_map(output, ground_truth)

    print(f"SSIM Value: {ssim_val.item():.4f}")

    # Plot
    fig, axs = plt.subplots(1, 2, figsize=(16, 4))
    axs[0].imshow(ground_truth.squeeze().cpu().numpy(), cmap='gray')
    axs[0].set_title('Ground Truth')
    axs[1].imshow(output.squeeze().cpu().numpy(), cmap='gray')
    axs[1].set_title('Model Output')

    for ax in axs:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    figs, axes = plt.subplots(1, 1, figsize=(4, 4))
    axes[0].imshow(ssim_map.squeeze().cpu().numpy(), cmap='gray')
    axes[0].set_title('SSIM Map')

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.show()
