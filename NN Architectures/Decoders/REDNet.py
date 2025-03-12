import torch
import torch.nn as nn
import torch.nn.functional as F

###############################################
# A Simple Residual Block for the Decoder
###############################################
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

###############################################
# REDNet Style Decoder
###############################################
class REDNetDecoder(nn.Module):
    def __init__(self, layersize, out_channels=1):
        """
        layersize: A list of channel dimensions for the encoder outputs, ordered from
                   deepest to shallowest. For example, for ResNet34 you might use:
                   [512, 256, 128, 64]
                   where 512 is the deepest feature map.
        out_channels: The number of channels for the final output.
        """
        super(REDNetDecoder, self).__init__()
        self.num_stages = len(layersize) - 1  # number of upsampling stages
        self.ups = nn.ModuleList()
        # In REDNet style, we add the skip connection (via element-wise addition)
        # so we need to ensure that the upsampled feature map and the skip feature have the same number of channels.
        # We assume layersize is already chosen so that the target upsample channels equal the skip's channels.
        self.res_blocks = nn.ModuleList()
        
        current_channels = layersize[0]
        for i in range(self.num_stages):
            target_channels = layersize[i+1]
            # Create upsampling layer: reduce channels from current to target.
            self.ups.append(
                nn.ConvTranspose2d(current_channels, target_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
            )
            # No adjustment is needed if target_channels already equals the skip connection channels.
            # Create a residual block to refine after the addition.
            self.res_blocks.append(ResidualBlock(target_channels))
            current_channels = target_channels
        
        # Final upscale to desired output resolution and a 1x1 conv for final output.
        self.final_upscale = nn.Upsample((128, 128), mode='bilinear', align_corners=False)
        self.out_conv = nn.Conv2d(layersize[-1], out_channels, kernel_size=1)
    
    def forward(self, *features):
        """
        Expects encoder outputs provided as arguments, ordered from shallowest to deepest.
        For example, for four features: s1 (shallow), s2, s3, s4 (deepest).
        The decoder reverses this order so that it starts from the deepest feature.
        """
        # Reverse the order: deepest first.
        skips = list(features)[::-1]
        x = skips[0]
        for i in range(self.num_stages):
            # Upsample the current feature map.
            x = self.ups[i](x)
            # Resize to match the spatial dimensions of the corresponding skip feature.
            x = F.interpolate(x, size=skips[i+1].shape[2:], mode="bilinear", align_corners=False)
            # Add (residual connection) the skip feature.
            x = x + skips[i+1]
            # Refine with a residual block.
            x = self.res_blocks[i](x)
        
        x = self.final_upscale(x)
        logits = self.out_conv(x)
        return torch.sigmoid(logits)
