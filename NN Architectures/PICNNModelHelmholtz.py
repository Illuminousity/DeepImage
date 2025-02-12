import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image

############################
# 1. PICNN MODEL DEFINITION
############################
class PICNN(nn.Module):
    def __init__(self):
        super(PICNN, self).__init__()
        
        # Encoder
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Decoder
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 1, kernel_size=3, padding=1)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        # Encoder
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        
        # Decoder
        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        x = self.deconv3(x)
        return x

############################
# 2. HELMHOLTZ-BASED LOSS
############################
class HelmholtzLoss(nn.Module):
    def __init__(self, wave_number=2 * 3.14159265359 / 660e-9, lambda_phys=0.1):
        super(HelmholtzLoss, self).__init__()
        self.wave_number = wave_number
        self.lambda_phys = lambda_phys

        # Discrete Laplacian kernel
        laplacian = torch.tensor(
            [[0,  1, 0],
             [1, -4, 1],
             [0,  1, 0]], dtype=torch.float32
        ).view(1,1,3,3)
        self.register_buffer("laplacian_kernel", laplacian)

    def forward(self, pred, target):
        # Ensure prediction and target have the same shape
        if pred.shape != target.shape:
            target = F.interpolate(target, size=pred.shape[2:], mode="bilinear", align_corners=False)
        
        # Data Fidelity (L1)
        data_loss = F.l1_loss(pred, target)
        
        # Helmholtz PDE Constraint
        lap_pred = F.conv2d(pred, self.laplacian_kernel, padding=1)
        helmholtz_residual = lap_pred + (self.wave_number**2)*pred
        pde_loss = F.mse_loss(helmholtz_residual, torch.zeros_like(helmholtz_residual))
        
        total_loss = data_loss + self.lambda_phys * pde_loss
        return total_loss



############################
# 3. FIXED DATASET
############################
class DiffusionDataset(Dataset):
    """

    Parses the index from 'diffused_imageXXX.png' and automatically
    looks for 'raw_imageXXX.png' in the Raw folder.
    """
    def __init__(self, diffused_dir, clean_dir, transform=None):
        super().__init__()
        self.diffused_dir = diffused_dir
        self.clean_dir = clean_dir
        self.transform = transform
        
        # We'll gather all filenames that match pattern "diffused_image(\d+).png"
        pattern = re.compile(r'^captured_frame_(\d+)\.png$')
        self.diffused_files = []
        
        for fname in os.listdir(self.diffused_dir):
            if pattern.match(fname):
                self.diffused_files.append(fname)
        
        # Sort by numeric index so data order is consistent
        self.diffused_files.sort(key=lambda x: int(pattern.match(x).group(1)))

    def __len__(self):
        return len(self.diffused_files)

    def __getitem__(self, idx):
        diffused_filename = self.diffused_files[idx]
        
        match = re.match(r'^captured_frame_(\d+)\.png$', diffused_filename)
        if not match:
            raise ValueError(
                f"File {diffused_filename} does not match 'diffused_image<number>.png' naming."
            )
        index_str = match.group(1)
        
        raw_filename = f"captured_frame_{index_str}.png"
        
        diffused_path = os.path.join(self.diffused_dir, diffused_filename)
        clean_path    = os.path.join(self.clean_dir, raw_filename)
        
        # Open images in grayscale
        diffused_img = Image.open(diffused_path).convert('L')
        clean_img    = Image.open(clean_path).convert('L')

        # Apply same transforms
        if self.transform:
            diffused_img = self.transform(diffused_img)
            clean_img    = self.transform(clean_img)

        return diffused_img, clean_img
