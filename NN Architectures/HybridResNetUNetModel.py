# Filename: SegmentResNetUNet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


########################################################################
# 1. ResNet34-based Encoder for Single-Channel Input
########################################################################
class ResNetEncoder(nn.Module):
    def __init__(self):
        """
        Builds a ResNet34 encoder from scratch (no pretrained weights),
        adapted to accept single-channel input.
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

########################################################################
# 2. U-Net Decoder
########################################################################
class UNetDecoder(nn.Module):
    def __init__(self, out_channels=1):
        super().__init__()
        # ResNet feature dims: s1=64, s2=64, s3=128, s4=256, s5=512

        self.up5   = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1)
        self.conv5 = nn.Conv2d(256+256, 256, 3, padding=1)

        self.up4   = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)
        self.conv4 = nn.Conv2d(128+128, 128, 3, padding=1)

        self.up3   = nn.ConvTranspose2d(128, 64,  3, stride=2, padding=1, output_padding=1)
        self.conv3 = nn.Conv2d(64+64,   64,  3, padding=1)

        self.up2   = nn.ConvTranspose2d(64,  64,  3, stride=2, padding=1, output_padding=1)
        self.conv2 = nn.Conv2d(64+64,   64,  3, padding=1)

        # Final 1-channel output (logits)
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.final_upscale = nn.Upsample((128,128))

    def forward(self, s1, s2, s3, s4, s5):
        x = self.up5(s5)
        x = F.interpolate(x, size=s4.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, s4], dim=1)
        x = F.relu(self.conv5(x))

        x = self.up4(x)
        x = F.interpolate(x, size=s3.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, s3], dim=1)
        x = F.relu(self.conv4(x))

        x = self.up3(x)
        x = F.interpolate(x, size=s2.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, s2], dim=1)
        x = F.relu(self.conv3(x))

        x = self.up2(x)
        x = F.interpolate(x, size=s1.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, s1], dim=1)
        x = F.relu(self.conv2(x))
        x = self.final_upscale(x)
        logits = self.out_conv(x)
        return logits

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        # pred, target: each [N, 1, H, W], with pred as probabilities in [0..1]
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() + self.eps
        dice = (2 * intersection) / union
        return 1 - dice

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, target):
        # logits [N,1,H,W], target [N,1,H,W]
        bce_val = self.bce(logits, target)
        pred_probs = torch.sigmoid(logits)
        dice_val = self.dice(pred_probs, target)
        return self.bce_weight * bce_val + self.dice_weight * dice_val



########################################################################
# 3. Combined Model
########################################################################
class ResNetUNetSegmentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ResNetEncoder()
        self.decoder = UNetDecoder(out_channels=1)

    def forward(self, x):
        s1, s2, s3, s4, s5 = self.encoder(x)
        logits = self.decoder(s1, s2, s3, s4, s5)
        return logits


########################################################################
# 4. Example Usage
########################################################################
if __name__ == "__main__":
    model = ResNetUNetSegmentation(weights=None)
    # Example input [batch=1, channels=1, H=128, W=128]
    sample_input = torch.randn(1, 1, 128, 128)
    out = model(sample_input)
    print("Output shape:", out.shape)  # [1,1,128,128]
    # Typically you'd feed these logits into BCEWithLogitsLoss
