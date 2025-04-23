import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b3

########################################################################
# EfficientNet-B3-based Encoder for Single-Channel Input
########################################################################
class EfficientNetEncoder(nn.Module):
    def __init__(self):
        """
        Builds an EfficientNet-B3 encoder adapted for single-channel input.
        Approximate feature channels:
          s1: [B, 40, H/2, W/2]
          s2: [B, 32, H/4, W/4]
          s3: [B, 48, H/8, W/8]
          s4: [B, 136, H/16, W/16]
          s5: [B, 384, H/32, W/32]
        """
        super().__init__()
        self.model = efficientnet_b3(weights=None)
        # Modify first convolution to accept 1-channel input.
        first_conv = self.model.features[0][0]
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=first_conv.out_channels,  # should be 40 for B3
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=False
        )
        nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
        self.model.features[0][0] = new_conv

    def forward(self, x):
        s1 = self.model.features[0](x)  # [B, 40, H/2, W/2]
        x = self.model.features[1](s1)
        s2 = self.model.features[2](x)  # [B, 32, H/4, W/4]
        s3 = self.model.features[3](s2)  # [B, 48, H/8, W/8]
        x = self.model.features[4](s3)
        s4 = self.model.features[5](x)  # [B, 136, H/16, W/16]
        x = self.model.features[6](s4)
        s5 = self.model.features[7](x)  # [B, 384, H/32, W/32]
        return s1, s2, s3, s4, s5