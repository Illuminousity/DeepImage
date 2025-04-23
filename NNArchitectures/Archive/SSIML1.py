import torch
import torch.nn as nn
import torch.nn.functional as F
import math

###############################
# 1. SSIM Utility Functions
###############################
def gaussian(window_size, sigma):
    # Use math.exp for float-based exponent
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

def ssim(img1, img2, window_size=11, sigma=1.5, val_range=1.0):
    # Expects img1, img2: (N, C, H, W)
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
# 2. Curriculum Loss
###############################
class L1SSIMLoss(nn.Module):
    """
    Gradually shift from L1-dominant to SSIM-dominant over training. This is because SSIM compares images, SSIM is more useful when the model can make better predictions
    alpha_start: The alpha at the beginning (more L1 if closer to 1.0).
    alpha_end: The alpha at the end (less L1 if lower).
    
    Loss formula: L = alpha * L1 + (1 - alpha) * (1 - SSIM)
    """
    def __init__(self, alpha_start=1.0, alpha_end=0.5, window_size=11, sigma=1.5, val_range=1.0):
        super().__init__()
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.alpha = alpha_start  # current alpha
        self.window_size = window_size
        self.sigma = sigma
        self.val_range = val_range
        self.l1_loss = nn.L1Loss()

    def set_epoch_fraction(self, fraction):
        """
        fraction: value in [0, 1] representing how far we are in training
        0 => alpha = alpha_start
        1 => alpha = alpha_end
        """
        self.alpha = self.alpha_start + (self.alpha_end - self.alpha_start) * fraction

    def forward(self, pred, target):
        # L1
        l1_val = self.l1_loss(pred, target)
        # SSIM
        ssim_val = ssim(pred, target, window_size=self.window_size, sigma=self.sigma, val_range=self.val_range)
        ssim_loss = 1.0 - ssim_val
        
        return self.alpha * l1_val + (1 - self.alpha) * ssim_loss
