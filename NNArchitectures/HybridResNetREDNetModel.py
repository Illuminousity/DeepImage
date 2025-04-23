# Filename: SegmentResNetUNet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from Encoders.ResNet import ResNetEncoder
from Decoders.REDNet import REDNetDecoder

########################################################################
# ResNet Encoder + U-Net Decoder - Layersize is defined by the chosen Encoder
########################################################################
class ResNetREDNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ResNetEncoder()
        self.decoder = REDNetDecoder(layersize=[512,256,128,64],out_channels=1)

    def forward(self, x):
        s1, s2, s3, s4, s5 = self.encoder(x)
        logits = self.decoder(s1, s2, s3, s4, s5)
        return logits


########################################################################
# Example Usage - Tests whether the class initializes correctly
########################################################################
if __name__ == "__main__":
    model = ResNetREDNet()
    # Example input [batch=1, channels=1, H=128, W=128]
    sample_input = torch.randn(1, 1, 128, 128)
    out = model(sample_input)
    print("Output shape:", out.shape)  # [1,1,128,128]
