import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image

############################
# 1. VGGNET-20 STYLE MODEL
############################
class VGGNet20(nn.Module):
    def __init__(self):
        super(VGGNet20, self).__init__()
        
        # Input Convolution
        self.input_conv = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        
        # Convolutional Blocks
        self.conv_layers = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(),
        )
        
        # Pooling & Reshape
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(512 * 128 * 96, 1024)
        self.fc2 = nn.Linear(1024, 512 * 128 * 96)
        
        # Upsampling & Output Layers
        self.upsample = nn.Upsample(size=(256, 192), mode="bilinear", align_corners=False)
        self.output_conv = nn.Conv2d(512, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # Input Layer
        x = F.relu(self.input_conv(x))
        
        # Convolution Blocks
        x = self.conv_layers(x)
        
        # Pooling
        x = self.pool(x)
        
        # Flatten & Fully Connected
        b, c, h, w = x.shape
        x = x.view(b, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.view(b, c, h, w)
        
        # Upsample & Output
        x = self.upsample(x)
        x = self.output_conv(x)
        
        return x

############################
# 2. HELMHOLTZ-BASED LOSS
############################
class HelmholtzLoss(nn.Module):
    def __init__(self, wave_number=1.0, lambda_phys=0.1):
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
    def __init__(self, diffused_dir, clean_dir, transform=None):
        super().__init__()
        self.diffused_dir = diffused_dir
        self.clean_dir = clean_dir
        self.transform = transform
        
        pattern = re.compile(r'^diffused_image(\d+)\.png$')
        self.diffused_files = []
        
        for fname in os.listdir(self.diffused_dir):
            if pattern.match(fname):
                self.diffused_files.append(fname)
        
        self.diffused_files.sort(key=lambda x: int(pattern.match(x).group(1)))

    def __len__(self):
        return len(self.diffused_files)

    def __getitem__(self, idx):
        diffused_filename = self.diffused_files[idx]
        
        match = re.match(r'^diffused_image(\d+)\.png$', diffused_filename)
        if not match:
            raise ValueError(
                f"File {diffused_filename} does not match 'diffused_image<number>.png' naming."
            )
        index_str = match.group(1)
        
        raw_filename = f"raw_image{index_str}.png"
        
        diffused_path = os.path.join(self.diffused_dir, diffused_filename)
        clean_path    = os.path.join(self.clean_dir, raw_filename)
        
        diffused_img = Image.open(diffused_path).convert('L')
        clean_img    = Image.open(clean_path).convert('L')

        if self.transform:
            diffused_img = self.transform(diffused_img)
            clean_img    = self.transform(clean_img)

        return diffused_img, clean_img
