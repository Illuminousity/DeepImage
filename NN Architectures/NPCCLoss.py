import torch
import torch.nn as nn

class NPCCLoss(nn.Module):
    def __init__(self):
        super(NPCCLoss, self).__init__()

    def forward(self, y, g):  # y = predicted, g = ground truth
        # Compute means
        y_mean = torch.mean(y, dim=[1, 2, 3], keepdim=True)
        g_mean = torch.mean(g, dim=[1, 2, 3], keepdim=True)

        # Center the data
        y_centered = y - y_mean
        g_centered = g - g_mean

        # Numerator: sum of element-wise product
        numerator = torch.sum(y_centered * g_centered, dim=[1, 2, 3])

        # Denominator: product of L2 norms
        denominator = torch.sqrt(
            torch.sum(y_centered ** 2, dim=[1, 2, 3]) *
            torch.sum(g_centered ** 2, dim=[1, 2, 3]) + 1e-8  # epsilon for stability
        )

        pcc = numerator / denominator  # Pearson correlation
        return -torch.mean(pcc)  # Negative Pearson = NPCC
