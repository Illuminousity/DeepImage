import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image

############################
# 1. OPTIMIZED VGGNET-20 STYLE MODEL
############################
class VGGNet20(nn.Module):
    def __init__(self):
        super(VGGNet20, self).__init__()
        
        # Input Convolution
        self.input_conv = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        
        # Convolutional Blocks (Reduced Feature Maps)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(),
        )
        
        # Pooling & Reshape (Adaptive Pooling to Reduce FC Layer Size)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: (batch, 256, 128, 96)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((8, 8))  # Output: (batch, 256, 8, 8)
        
        # Fully Connected Layers (Reduced Size)
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 256 * 8 * 8)
        
        # Upsampling & Output Layers
        self.upsample = nn.Upsample(size=(192, 256), mode="bilinear", align_corners=False)
        self.output_conv = nn.Conv2d(256, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # Input Layer
        x = F.relu(self.input_conv(x))
        
        # Convolution Blocks
        x = self.conv_layers(x)
        
        # Pooling
        x = self.pool(x)  # Output: (batch, 256, 128, 96)
        x = self.global_avg_pool(x)  # Output: (batch, 256, 8, 8)
        
        # Flatten & Fully Connected
        b, c, h, w = x.shape  # Expecting (batch, 256, 8, 8)
        x = x.view(b, -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.view(b, 256, 8, 8)  # Reshape correctly
        
        # Upsample & Output
        x = self.upsample(x)  # (batch, 256, 192, 256)
        x = self.output_conv(x)
        
        return x

############################
# 2. Physics-BASED LOSS
############################
class SpecklePhysicsLoss(nn.Module):
    def __init__(self, wave_number=(2 * 3.14159265359 / 660e-9), lambda_helm=0, lambda_speckle=0, lambda_fourier=0):
        super(SpecklePhysicsLoss, self).__init__()
        self.wave_number = wave_number
        self.lambda_helm = lambda_helm
        self.lambda_speckle = lambda_speckle
        self.lambda_fourier = lambda_fourier

        # Laplacian Kernel (Finite Difference Approximation)
        laplacian = torch.tensor(
            [[0, 1, 0],
             [1, -4, 1],
             [0, 1, 0]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer("laplacian_kernel", laplacian)

    def forward(self, pred, target):
        # Ensure shape consistency
        if pred.shape != target.shape:
            target = F.interpolate(target, size=pred.shape[2:], mode="bilinear", align_corners=False)

        # Standard Reconstruction Loss (L1 loss)
        data_loss = F.l1_loss(pred, target)


        # Speckle Statistics Loss
        mean_pred = torch.mean(pred)
        var_pred = torch.var(pred)
        speckle_loss = F.mse_loss(var_pred / (mean_pred**2), torch.tensor(1.0, device=pred.device))

        # Fourier Spectrum Loss
        pred_fft = torch.fft.fft2(pred)
        target_fft = torch.fft.fft2(target)
        fourier_loss = F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))

        # Total loss
        total_loss = data_loss +  self.lambda_speckle * speckle_loss + self.lambda_fourier * fourier_loss

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
        
        pattern = re.compile(r'^captured_frame_(\d+)\.png$')
        self.diffused_files = []
        
        for fname in os.listdir(self.diffused_dir):
            if pattern.match(fname):
                self.diffused_files.append(fname)
        
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
        
        diffused_img = Image.open(diffused_path).convert('L')
        clean_img    = Image.open(clean_path).convert('L')

        if self.transform:
            diffused_img = self.transform(diffused_img)
            clean_img    = self.transform(clean_img)

        return diffused_img, clean_img
