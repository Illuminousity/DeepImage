import torch
import torch.nn as nn
import torch.nn.functional as F

########################################################################
# Attention Gate Module
########################################################################
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        """
        Parameters:
          F_g: Number of channels in the gating (decoder) signal.
          F_l: Number of channels in the skip connection from the encoder.
          F_int: Number of intermediate channels (typically half of F_g).
        """
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
        # g: gating signal (from decoder)
        # x: skip connection (from encoder)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

########################################################################
# U-Net Decoder with Attention Gates (Dynamic Version)
########################################################################
class UNetDecoder(nn.Module):
    def __init__(self, layersize, out_channels=1):
        """
        layersize: A list of channel dimensions for the encoder outputs in *decreasing* order.
                   For example, for EfficientNet-B3 you might pass: [384, 136, 48, 32, 40]
                   where 384 is the deepest feature map (s5) and 40 the shallowest (s1).
        out_channels: The number of channels for the final output.
        """
        super(UNetDecoder, self).__init__()
        
        # The number of upsampling stages is one less than the number of skip connections.
        self.num_stages = len(layersize) - 1
        
        # We'll build up the decoder using ModuleLists.
        self.ups = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.conv_blocks = nn.ModuleList()
        
        # Start with the deepest encoder output.
        current_channels = layersize[0]
        for i in range(self.num_stages):
            # For stages 0 to num_stages-2, we set the upsample output channels to half the input.
            # For the final stage, we set it to match the shallow skip channel.
            if i < self.num_stages - 1:
                out_ch = current_channels // 2
            else:
                out_ch = layersize[-1]
            
            # Create upsampling layer.
            self.ups.append(
                nn.ConvTranspose2d(current_channels, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1)
            )
            # Create attention gate for the corresponding skip connection.
            # The gating signal has out_ch channels and the skip connection has layersize[i+1] channels.
            F_g = out_ch
            F_l = layersize[i+1]
            F_int = F_g // 2 if F_g // 2 > 0 else 1
            self.attentions.append(AttentionGate(F_g, F_l, F_int))
            # Create the convolutional block after concatenation.
            # Input channels = upsampled channels (out_ch) + skip channels (F_l).
            self.conv_blocks.append(
                nn.Sequential(
                    nn.Conv2d(out_ch + F_l, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                )
            )
            # Update current_channels for the next stage.
            current_channels = out_ch
        
        # Final upscale to desired output resolution and a 1x1 conv for final logits.
        self.final_upscale = nn.Upsample((128, 128), mode='bilinear', align_corners=False)
        self.out_conv = nn.Conv2d(layersize[-1], out_channels, kernel_size=1)
    
    def forward(self, s1, s2, s3, s4, s5):
        """
        Expects encoder outputs in the order:
          s1: shallowest (e.g., 40 channels)
          s2: (e.g., 32 channels)
          s3: (e.g., 48 channels)
          s4: (e.g., 136 channels)
          s5: deepest (e.g., 384 channels)
        For the decoder we reverse the order so that we start with s5.
        """
        # Reverse the order: deepest first.
        skips = [s5, s4, s3, s2, s1]
        x = skips[0]
        for i in range(self.num_stages):
            x = self.ups[i](x)
            # Resize to match the spatial dimensions of the corresponding skip.
            x = F.interpolate(x, size=skips[i+1].shape[2:], mode="bilinear", align_corners=False)
            # Apply attention on the skip connection.
            skip_att = self.attentions[i](g=x, x=skips[i+1])
            # Concatenate the upsampled feature map with the attended skip connection.
            x = torch.cat([x, skip_att], dim=1)
            # Process through the conv block.
            x = self.conv_blocks[i](x)
        
        x = self.final_upscale(x)
        logits = self.out_conv(x)
        return torch.sigmoid(logits)
