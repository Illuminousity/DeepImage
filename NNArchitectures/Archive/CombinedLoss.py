import torch
import torch.nn as nn
import torch.nn.functional as F

from NPCCLoss import NPCCLoss  # Make sure NPCCLoss.py is in the same directory or installed

############################################
# Simple SSIM for Loss (PyTorch)
############################################
class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        

    def gaussian_window(self, channel, window_size, sigma=1.5):
        gauss = torch.Tensor([
            (-(x - window_size//2)**2) / float(2*sigma*sigma)
            for x in range(window_size)
        ])
        gauss = torch.exp(gauss)
        gauss = gauss / gauss.sum()
        _1D_window = gauss.unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float()
        window = _2D_window.unsqueeze(0).unsqueeze(0)
        window = window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def ssim(self, img1, img2, window_size=11, window=None, size_average=True, C1=0.01**2, C2=0.03**2):
        # Modified from a typical PyTorch SSIM snippet.
        _, channel, height, width = img1.size()

        if window is None:
            real_size = min(window_size, height, width)
            window = self.gaussian_window(channel, real_size)
            window = window.to(img1.device)
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map

    def forward(self, img1, img2):
        if img1.dim() < 4:
            img1 = img1.unsqueeze(0)
        if img2.dim() < 4:
            img2 = img2.unsqueeze(0)
        return 1.0 - self.ssim(img1, img2, window_size=self.window_size, size_average=self.size_average)

#######################################################
# CombinedLoss merges NPCC, L1, and SSIM
#######################################################

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.4, beta=0.3, gamma=0.3):
        """
        alpha -> weight for NPCC (which is negative correlation so smaller=better)
        beta  -> weight for L1
        gamma -> weight for SSIM (or rather 1-SSIM)
        """
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.npcc = NPCCLoss()
        self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss()

    def forward(self, pred, target):
        # npcc already returns a negative correlation => best is -1
        npcc_val = self.npcc(pred, target)
        # l1 best is 0
        l1_val = self.l1(pred, target)
        # ssim returns (1- ssim) => best is 0
        ssim_val = self.ssim(pred, target)

        combined = self.alpha * npcc_val + self.beta * l1_val + self.gamma * ssim_val
        return combined

if __name__ == "__main__":
    # Quick test
    model_loss = CombinedLoss(alpha=0.4, beta=0.3, gamma=0.3)

    pred = torch.rand(2, 1, 128, 128)
    tgt = torch.rand(2, 1, 128, 128)

    loss_val = model_loss(pred, tgt)
    print("Combined Loss:", loss_val.item())
