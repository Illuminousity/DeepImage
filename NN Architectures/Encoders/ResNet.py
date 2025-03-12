import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

########################################################################
# ResNet34-based Encoder
########################################################################
class ResNetEncoder(nn.Module):
    def __init__(self):
        """
        Builds a ResNet34 encoder adapted to accept single-channel input (Greyscale).
        """
        super().__init__()
        # 1) Create resnet34 with no pretrained weights
        resnet = models.resnet34(weights=None)

        # 2) Replace the very first conv layer to accept 1 channel instead of 3
        #    We'll use the same kernel size/stride/padding as standard ResNet.
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=64,
            kernel_size=7, stride=2, padding=3, bias=False
        )

        # 3) Manually re-initialize weights (Kaiming is typical for ResNets).
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')

        # 4) Extract remaining layers from the standard resnet, ignoring the original conv1.
        #    The children order is: [conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc]
        #    We'll skip avgpool and fc because we only need encoder stages.
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1  # 64 channels
        self.layer2 = resnet.layer2  # 128 channels
        self.layer3 = resnet.layer3  # 256 channels
        self.layer4 = resnet.layer4  # 512 channels

    def forward(self, x):
        # Stage 1: conv1 + bn + relu + maxpool
        x = self.conv1(x)          # [B, 64,  H/2,  W/2]
        x = self.bn1(x)
        x = self.relu(x)
        s1 = x                     # Keep for skip-connection if desired
        x = self.maxpool(x)        # [B, 64,  H/4,  W/4]

        # Stage 2
        x = self.layer1(x)         # -> [B, 64,  H/4,  W/4]
        s2 = x

        # Stage 3
        x = self.layer2(x)         # -> [B, 128, H/8,  W/8]
        s3 = x

        # Stage 4
        x = self.layer3(x)         # -> [B, 256, H/16, W/16]
        s4 = x

        # Stage 5
        x = self.layer4(x)         # -> [B, 512, H/32, W/32]
        s5 = x

        # Return intermediate features for U-Net style decoding
        return s1, s2, s3, s4, s5