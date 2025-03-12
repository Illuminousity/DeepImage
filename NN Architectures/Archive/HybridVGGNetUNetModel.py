import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


########################################################################
# 1. Custom VGG16-based Encoder for Single-Channel Input
########################################################################
class VGGEncoder(nn.Module):
    def __init__(self):
        """
        Builds a VGG16 encoder from scratch (no pretrained weights),
        adapted to accept single-channel input.
        We will store intermediate features at the end of each block
        (conv -> conv -> pool) to use as skip connections.
        """
        super().__init__()
        # 1) Load a plain VGG16 (no pretrained weights)
        #    The 'features' portion has all convolution and pooling layers.
        #    For a single-channel input, we must replace the first conv layer.
        vgg = models.vgg16(weights=None)
        
        # 2) Extract the feature layers
        features = list(vgg.features)

        # 3) Replace the first Conv2d (index 0) to accept 1 input channel
        #    Original VGG16 conv1 is:
        #    Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        in_channels_original = features[0].in_channels  # Should be 3
        out_channels_original = features[0].out_channels  # Should be 64
        kernel_size = features[0].kernel_size  # (3,3)
        stride = features[0].stride  # (1,1)
        padding = features[0].padding  # (1,1)

        new_conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=out_channels_original,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True
        )
        # Optionally re-init weights (Kaiming, xavier, etc.)
        nn.init.kaiming_normal_(new_conv1.weight, mode='fan_out', nonlinearity='relu')
        features[0] = new_conv1

        # 4) Now we register the updated feature list as a nn.Sequential
        self.vgg_features = nn.Sequential(*features)
        
        # We'll collect feature maps at the following indices:
        # VGG16 has conv blocks (assuming each block is conv->conv->pool except block3 has conv->conv->conv->pool, etc.)
        # Indices of maxpool layers are 4, 9, 16, 23, 30 in vgg.features for standard VGG16:
        #    block1: 0,1,2,3 -> pool at index 4
        #    block2: 5,6,7,8 -> pool at index 9
        #    block3: 10,11,12,13,14 -> pool at index 16
        #    block4: 17,18,19,20,21 -> pool at index 23
        #    block5: 24,25,26,27,28 -> pool at index 30
        # We will store the outputs after each pool for skip connections in U-Net.
        # If you want “finer” or “coarser” skip connections, adjust these indices as desired.

        self.pool_indices = [4, 9, 16, 23, 30]  # output indices after each pool layer

    def forward(self, x):
        """
        Returns five intermediate feature maps:
        s1,s2,s3,s4,s5 for use by the UNet decoder.
        """
        features = []
        for idx, layer in enumerate(self.vgg_features):
            x = layer(x)
            if idx in self.pool_indices:
                # store the output at the end of each block
                features.append(x)

        # The five stored features correspond to:
        # s1 -> after block1 (64 channels)
        # s2 -> after block2 (128 channels)
        # s3 -> after block3 (256 channels)
        # s4 -> after block4 (512 channels)
        # s5 -> after block5 (512 channels)
        s1, s2, s3, s4, s5 = features
        return s1, s2, s3, s4, s5


########################################################################
# 2. UNet Decoder
########################################################################
class UNetDecoder(nn.Module):
    def __init__(self, out_channels=1):
        """
        We match the channel counts from the VGG blocks:
          s1=64, s2=128, s3=256, s4=512, s5=512
        """
        super().__init__()

        # Up from s5=512 + s4=512 -> 512
        self.up5   = nn.ConvTranspose2d(512, 512, 3, stride=2, padding=1, output_padding=1)
        self.conv5 = nn.Conv2d(512 + 512, 512, 3, padding=1)

        # Up from s4=512 + s3=256 -> 256
        self.up4   = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1)
        self.conv4 = nn.Conv2d(256 + 256, 256, 3, padding=1)

        # Up from s3=256 + s2=128 -> 128
        self.up3   = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)
        self.conv3 = nn.Conv2d(128 + 128, 128, 3, padding=1)

        # Up from s2=128 + s1=64 -> 64
        self.up2   = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.conv2 = nn.Conv2d(64 + 64, 64, 3, padding=1)

        # We may optionally do a final upsample back to the input shape.
        # For example, if the input was 128x128, VGG downsampled 5 times
        # yields 4x4 at the deepest level (roughly). We'll do a last upsample
        # to 128x128, then 1x1 conv for the segmentation map.
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.final_upscale = nn.Upsample((128, 128), mode='bilinear', align_corners=False)

    def forward(self, s1, s2, s3, s4, s5):
        # s5 up + skip with s4
        x = self.up5(s5)
        x = F.interpolate(x, size=s4.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, s4], dim=1)
        x = F.relu(self.conv5(x))

        # s4 up + skip with s3
        x = self.up4(x)
        x = F.interpolate(x, size=s3.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, s3], dim=1)
        x = F.relu(self.conv4(x))

        # s3 up + skip with s2
        x = self.up3(x)
        x = F.interpolate(x, size=s2.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, s2], dim=1)
        x = F.relu(self.conv3(x))

        # s2 up + skip with s1
        x = self.up2(x)
        x = F.interpolate(x, size=s1.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, s1], dim=1)
        x = F.relu(self.conv2(x))

        # final up to (128,128) and 1x1 conv for segmentation
        x = self.final_upscale(x)
        logits = self.out_conv(x)
        return logits


########################################################################
# 3. Example Losses: Dice and BCE+Dice
########################################################################
class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        # pred, target: each [N, 1, H, W], with pred in [0..1]
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() + self.eps
        dice = 2.0 * intersection / union
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
# 4. Combined Model
########################################################################
class VGGUNetSegmentation(nn.Module):
    def __init__(self):
        """
        Full segmentation model that uses the VGGEncoder and
        the UNetDecoder from above to predict a 1-channel mask.
        """
        super().__init__()
        self.encoder = VGGEncoder()
        self.decoder = UNetDecoder(out_channels=1)

    def forward(self, x):
        # x: [N,1,H,W]
        s1, s2, s3, s4, s5 = self.encoder(x)
        logits = self.decoder(s1, s2, s3, s4, s5)
        return logits


########################################################################
# 5. Example Usage
########################################################################
if __name__ == "__main__":
    model = VGGUNetSegmentation()
    sample_input = torch.randn(1, 1, 128, 128)
    out = model(sample_input)
    print("Output shape:", out.shape)  # [1, 1, 128, 128]
