import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

# ResNet-based Encoder with Spatial Preservation
class ResNetEncoder(nn.Module):
    def __init__(self, weights='IMAGENET1K_V1'):
        super(ResNetEncoder, self).__init__()
        resnet = models.resnet34(weights=weights)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)  # Reduce stride to preserve spatial details
        self.encoder_layers = nn.Sequential(*list(resnet.children())[:-2])  # Remove FC and avg pooling layers
    
    def forward(self, x):
        return self.encoder_layers(x)

# U-Net style Decoder with Additional Convolutions
class UNetDecoder(nn.Module):
    def __init__(self, num_classes=1):
        super(UNetDecoder, self).__init__()
        self.upsample1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.upsample2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.upsample3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.upsample4 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        
        self.final_upsample = nn.Upsample(size=(128, 128), mode="bilinear", align_corners=True)
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.upsample1(x)
        x = self.relu1(self.conv1(x))
        
        x = self.upsample2(x)
        x = self.relu2(self.conv2(x))
        
        x = self.upsample3(x)
        x = self.relu3(self.conv3(x))
        
        x = self.upsample4(x)
        x = self.relu4(self.conv4(x))
        
        x = self.final_upsample(x)
        x = self.final_conv(x)
        return x

# Combined ResNet + U-Net Hybrid Model
class ResNetUNet(nn.Module):
    def __init__(self, num_classes=1, weights='IMAGENET1K_V1'):
        super(ResNetUNet, self).__init__()
        self.encoder = ResNetEncoder(weights=weights)
        self.decoder = UNetDecoder(num_classes=num_classes)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Physics-Based Loss Functions
def helmholtz_loss(output, wavelength=660e-9):
    laplacian = torch.nn.functional.laplace(output)
    k = 2 * np.pi / wavelength  # Wave number
    loss = torch.mean((laplacian + k**2 * output) ** 2)
    print(f"Helmholtz Loss: {loss.item()}")  # Debug print
    return loss

def fourier_loss(output, target):
    output_fft = torch.fft.fft2(output)
    target_fft = torch.fft.fft2(target)
    loss = torch.mean((torch.abs(output_fft) - torch.abs(target_fft)) ** 2)
    print(f"Fourier Loss: {loss.item()}")  # Debug print
    return loss

def physics_loss(output, target, lambda_helm=0.05, lambda_fourier=0.01):
    mse = torch.nn.functional.mse_loss(output, target)
    helm = helmholtz_loss(output)
    fourier = fourier_loss(output, target)
    total_loss = mse + lambda_helm * helm + lambda_fourier * fourier
    print(f"MSE Loss: {mse.item()}, Total Loss: {total_loss.item()}")  # Debug print
    return total_loss

# Example usage
if __name__ == "__main__":
    model = ResNetUNet(num_classes=1)
    sample_input = torch.randn(1, 1, 128, 128)  # Example input tensor (grayscale)
    output = model(sample_input)
    print("Output shape:", output.shape)  # Should match input spatial dimensions
