import torch
import torch.nn as nn
import torch.nn.functional as F



############################
# 1. OPTIMIZED VGGNET-20 STYLE MODEL
############################

class VGGNet20(nn.Module):
    def __init__(self):
        super(VGGNet20, self).__init__()

        # Input Convolution
        self.input_conv = nn.Conv2d(1, 64, kernel_size=3, padding=1)

        # Convolutional Blocks with Batch Normalization
        self.conv_layers = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(512),
        )

        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: (batch, 512, 16, 12)

        # 1x1 Convolutions Instead of Fully Connected Layers
        self.fc1 = nn.Conv2d(512, 1024, kernel_size=1)
        self.fc2 = nn.Conv2d(1024, 512, kernel_size=1)

        # Upsampling with Bilinear Interpolation
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Output Convolution
        self.output_conv = nn.Conv2d(512, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.input_conv(x))
        x = self.conv_layers(x)
        x = self.pool(x)  # Output after pooling


        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x = self.upsample(x)
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
        print("Data loss:", data_loss.item())

        # Speckle Statistics Loss
        mean_pred = torch.mean(pred)
        var_pred = torch.var(pred)
        speckle_loss = F.mse_loss(var_pred / (mean_pred**2), torch.tensor(1.0, device=pred.device))
        print("Speckle loss:", speckle_loss.item())

        # Fourier Spectrum Loss
        pred_fft = torch.fft.fft2(pred)
        target_fft = torch.fft.fft2(target)
        fourier_loss = (F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft)))/ (pred.shape[2] * pred.shape[3])
        print("Fourier loss:", fourier_loss.item())
        

        # Total loss
        total_loss = data_loss + self.lambda_speckle * speckle_loss + self.lambda_fourier * fourier_loss

        return total_loss



