import torch
import torch.nn as nn
import torch.nn.functional as F

###############################################
# Attention Gate Module
###############################################
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi  # Weighted skip connection

###############################################
# Improved Residual Block (Deeper + Dilated Conv)
###############################################
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, dilation=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2)
        self.bn2 = nn.BatchNorm2d(channels)
        
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=3, dilation=3)
        self.bn3 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        out += residual  # Element-wise addition for residual connection
        out = self.relu(out)
        return out

###############################################
# Enhanced REDNet Decoder with Attention & SE
###############################################
class REDNetDecoder(nn.Module):
    def __init__(self, layersize, out_channels=1):
        super(REDNetDecoder, self).__init__()
        self.num_stages = len(layersize) - 1  # Number of upsampling stages
        self.ups = nn.ModuleList()
        self.attentions = nn.ModuleList()  # Attention Gates
        self.res_blocks = nn.ModuleList()
        
        current_channels = layersize[0]
        for i in range(self.num_stages):
            target_channels = layersize[i+1]
            
            # Create upsampling layer
            self.ups.append(
                nn.ConvTranspose2d(current_channels, target_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
            )
            
            # Add attention gate for skip connections
            self.attentions.append(AttentionGate(target_channels, target_channels, target_channels // 2))
            
            # Create deeper residual block with dilated convolutions
            self.res_blocks.append(ResidualBlock(target_channels))
            
            current_channels = target_channels
        
        # Final upsampling to desired output size
        self.final_upscale = nn.Upsample((128, 128), mode='bilinear', align_corners=False)
        self.out_conv = nn.Conv2d(layersize[-1], out_channels, kernel_size=1)

    def forward(self, *features):
        skips = list(features)[::-1]  # Reverse encoder outputs (deepest to shallowest)
        x = skips[0]  # Start from the deepest feature map
        
        for i in range(self.num_stages):
            x = self.ups[i](x)  # Upsample feature map
            x = F.interpolate(x, size=skips[i+1].shape[2:], mode="bilinear", align_corners=False)
            
            # Apply attention to skip connection before adding
            skip_att = self.attentions[i](g=x, x=skips[i+1])
            x = x + skip_att  # Element-wise addition (like residual fusion)
            
            # Process through residual block
            x = self.res_blocks[i](x)
        
        x = self.final_upscale(x)
        logits = self.out_conv(x)
        return torch.sigmoid(logits)
