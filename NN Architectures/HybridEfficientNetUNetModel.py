import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0

########################################################################
# 1. EfficientNet-B0-based Encoder for Single-Channel Input
########################################################################
class EfficientNetEncoder(nn.Module):
    def __init__(self):
        """
        Builds an EfficientNet-B0 encoder from torchvision (no pretrained weights),
        adapted to accept single-channel input and output intermediate feature maps.
        
        Feature extraction:
          s1: Output of stem (features[0]) - [B, 32, H/2, W/2]
          s2: Output of block 2 (features[2]) - [B, 24, H/4, W/4]
          s3: Output of block 3 (features[3]) - [B, 40, H/8, W/8]
          s4: Output of block 5 (features[5]) - [B, 112, H/16, W/16]
          s5: Output of block 7 (features[7]) - [B, 320, H/32, W/32]
        """
        super().__init__()
        # Create EfficientNet-B0 with no pretrained weights
        self.model = efficientnet_b0(weights=None)
        
        # Modify the first conv layer (in features[0]) to accept 1 channel instead of 3.
        # self.model.features[0] is a ConvBnAct module; we replace its conv layer.
        first_conv = self.model.features[0][0]
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=False
        )
        nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
        self.model.features[0][0] = new_conv

    def forward(self, x):
        # s1: after stem, index 0 (ConvBnAct) => [B, 32, H/2, W/2]
        s1 = self.model.features[0](x)
        # Block 1: index 1 (MBConv, stride=1) – not used as a skip
        x = self.model.features[1](s1)
        # s2: Block 2, index 2 (MBConv, stride=2) => [B, 24, H/4, W/4]
        s2 = self.model.features[2](x)
        # s3: Block 3, index 3 (MBConv, stride=2) => [B, 40, H/8, W/8]
        s3 = self.model.features[3](s2)
        # Process blocks 4 and 5: 
        # Block 4, index 4 (MBConv, stride=2) => intermediate output [B, 80, H/16, W/16]
        x = self.model.features[4](s3)
        # s4: Block 5, index 5 (MBConv, stride=1) => [B, 112, H/16, W/16]
        s4 = self.model.features[5](x)
        # Process blocks 6 and 7:
        # Block 6, index 6 (MBConv, stride=2) => intermediate output [B, 192, H/32, W/32]
        x = self.model.features[6](s4)
        # s5: Block 7, index 7 (MBConv, stride=1) => [B, 320, H/32, W/32]
        s5 = self.model.features[7](x)
        return s1, s2, s3, s4, s5

########################################################################
# 2. U-Net Decoder (Modified with BatchNorm for Improved Performance)
########################################################################
class UNetDecoder(nn.Module):
    def __init__(self, out_channels=1):
        """
        Decoder adjusted for the channel dimensions of EfficientNet-B0 features:
          s1: 32, s2: 24, s3: 40, s4: 112, s5: 320.
        We upsample and then concatenate the corresponding skip features.
        """
        super().__init__()
        # Decoder stage 1: upsample from s5 (320) to combine with s4 (112)
        self.up5 = nn.ConvTranspose2d(320, 160, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv5 = nn.Sequential(
            nn.Conv2d(160 + 112, 160, kernel_size=3, padding=1),
            nn.BatchNorm2d(160),
            nn.ReLU(inplace=True)
        )
        
        # Decoder stage 2: upsample from 160 to combine with s3 (40)
        self.up4 = nn.ConvTranspose2d(160, 80, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv4 = nn.Sequential(
            nn.Conv2d(80 + 40, 80, kernel_size=3, padding=1),
            nn.BatchNorm2d(80),
            nn.ReLU(inplace=True)
        )
        
        # Decoder stage 3: upsample from 80 to combine with s2 (24)
        self.up3 = nn.ConvTranspose2d(80, 40, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv3 = nn.Sequential(
            nn.Conv2d(40 + 24, 40, kernel_size=3, padding=1),
            nn.BatchNorm2d(40),
            nn.ReLU(inplace=True)
        )
        
        # Decoder stage 4: upsample from 40 to combine with s1 (32)
        self.up2 = nn.ConvTranspose2d(40, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32 + 32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Optional final upscale and 1x1 conv to produce the segmentation logits.
        self.final_upscale = nn.Upsample((128,128))
        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)
    
    def forward(self, s1, s2, s3, s4, s5):
        x = self.up5(s5)
        x = F.interpolate(x, size=s4.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, s4], dim=1)
        x = self.conv5(x)
        
        x = self.up4(x)
        x = F.interpolate(x, size=s3.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, s3], dim=1)
        x = self.conv4(x)
        
        x = self.up3(x)
        x = F.interpolate(x, size=s2.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, s2], dim=1)
        x = self.conv3(x)
        
        x = self.up2(x)
        x = F.interpolate(x, size=s1.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, s1], dim=1)
        x = self.conv2(x)
        
        x = self.final_upscale(x)
        logits = self.out_conv(x)
        return logits

########################################################################
# 3. Combined Model: EfficientNet Encoder + U-Net Decoder
########################################################################
class EfficientNetUNetSegmentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = EfficientNetEncoder()
        self.decoder = UNetDecoder(out_channels=1)
    
    def forward(self, x):
        s1, s2, s3, s4, s5 = self.encoder(x)
        logits = self.decoder(s1, s2, s3, s4, s5)
        return logits

########################################################################
# 4. Example Usage
########################################################################
if __name__ == "__main__":
    model = EfficientNetUNetSegmentation()
    # Example input [batch=1, channels=1, H=128, W=128]
    sample_input = torch.randn(1, 1, 128, 128)
    out = model(sample_input)
    print("Output shape:", out.shape)  # Typically [1,1,128,128]
