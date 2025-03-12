import torch
import torch.nn as nn
from Encoders.EfficientNet import EfficientNetEncoder
from Decoders.REDNet import REDNetDecoder




########################################################################
# EfficientNet Encoder + U-Net Decoder 
########################################################################
class EfficientNetREDNet(nn.Module):
    def __init__(self):
        super(EfficientNetREDNet, self).__init__()
        self.encoder = EfficientNetEncoder()
        self.decoder = REDNetDecoder(layersize=[384, 136, 48, 32, 40],out_channels=1)
    
    def forward(self, x):
        s1, s2, s3, s4, s5 = self.encoder(x)
        logits = self.decoder(s1, s2, s3, s4, s5)
        return logits

########################################################################
# Example Usage
########################################################################
if __name__ == "__main__":
    model = EfficientNetREDNet()
    # Example input: batch size 1, single-channel, 128x128 image
    sample_input = torch.randn(1, 1, 128, 128)
    out = model(sample_input)
    print("Output shape:", out.shape)  # Expected output shape: [1, 1, 128, 128]
