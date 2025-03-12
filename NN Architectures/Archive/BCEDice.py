import torch
import torch.nn as nn
import torch.nn.functional as F

class BCEWithDiceLoss(nn.Module):
    def __init__(self, weight_bce=0.5, weight_dice=0.5, smooth=1):
        """
        Initializes the combined BCE and Dice loss.
        
        Args:
            weight_bce (float): Weight for the BCE loss.
            weight_dice (float): Weight for the Dice loss.
            smooth (float): Smoothing factor to prevent division by zero.
        """
        super(BCEWithDiceLoss, self).__init__()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        self.smooth = smooth
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        # Compute BCE loss using logits (BCEWithLogitsLoss applies sigmoid internally)
        bce = self.bce_loss(logits, targets)
        
        # Apply sigmoid to logits to get probabilities for Dice loss
        probs = torch.sigmoid(logits)
        
        # Flatten tensors to compute Dice score over all pixels
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        # Compute the intersection and Dice coefficient
        intersection = (probs_flat * targets_flat).sum()
        dice_score = (2 * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        dice_loss = 1 - dice_score
        
        # Return weighted sum of BCE and Dice losses
        return self.weight_bce * bce + self.weight_dice * dice_loss
