import torch
import torch.nn as nn
import torch.fft as fft

class FourierLoss(nn.Module):
    def __init__(self, alpha=0.5):
        """
        Custom loss function that combines L1 loss and Fourier loss.
        :param alpha: Weighting factor for balancing L1 loss and Fourier loss.
        """
        super(FourierLoss, self).__init__()
        self.alpha = alpha
        self.l1_loss = nn.L1Loss()

    def forward(self, outputs, targets):
        """
        Compute the combined loss.
        """
        # Compute L1 loss
        l1_loss = self.l1_loss(outputs, targets)
        
        # Compute Fourier transform of outputs and targets
        outputs_fft = fft.fft2(outputs)
        targets_fft = fft.fft2(targets)
        
        # Compute magnitude difference in Fourier domain
        fourier_loss = self.l1_loss(torch.abs(outputs_fft), torch.abs(targets_fft))
        
        # Weighted sum of L1 loss and Fourier loss
        total_loss = (1 - self.alpha) * l1_loss + self.alpha * fourier_loss
        return total_loss