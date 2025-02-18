# Filename: DebugTrain.py

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import matplotlib.pyplot as plt

from HybridResNetUNetModel import ResNetUNetSegmentation, BCEDiceLoss
from DiffusionDataset import DiffusionDataset  # your custom dataset

def debug_train_model(model, dataloader, device, num_epochs=50):
    model.train()
    # Use BCEWithLogitsLoss for binary segmentation
    criterion = BCEDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    losses = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for diffused, gt_mask in dataloader:
            diffused, gt_mask = diffused.to(device), gt_mask.to(device)

            # Forward
            logits = model(diffused)
            # BCEWithLogitsLoss expects logits in [N,1,H,W] and gt in [N,1,H,W]
            # gt should be {0,1} or in [0,1].
            loss = criterion(logits, gt_mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        epoch_loss = running_loss / len(dataloader)
        losses.append(epoch_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    plt.figure(figsize=(7,5))
    plt.plot(range(1, num_epochs+1), losses, marker='o')
    plt.title("Debug Overfit on 1-5 Samples")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

def visualize_predictions(model, dataloader, device):
    import matplotlib.pyplot as plt
    
    model.eval()
    with torch.no_grad():
        for diffused, gt_mask in dataloader:
            diffused = diffused.to(device)
            gt_mask = gt_mask.cpu().numpy()  # shape [N,1,H,W]
            
            logits = model(diffused)
            preds = torch.sigmoid(logits).cpu().numpy()  # shape [N,1,H,W]

            # Display the first sample in the batch
            diffused_img = diffused[0].cpu().squeeze().numpy()  # [H,W]
            pred_mask = preds[0,0,:,:]  # [H,W]
            true_mask = gt_mask[0,0,:,:]

            fig, axes = plt.subplots(1, 3, figsize=(10,3))
            axes[0].imshow(diffused_img, cmap='gray')
            axes[0].set_title("Diffused Input")
            axes[1].imshow(pred_mask, cmap='gray', vmin=0, vmax=1)
            axes[1].set_title("Predicted Mask")
            axes[2].imshow(true_mask, cmap='gray', vmin=0, vmax=1)
            axes[2].set_title("Ground Truth")

            for ax in axes:
                ax.axis("off")
            plt.show()
            break  # just show one batch

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetUNetSegmentation().to(device)

    # Load the full dataset
    transform = transforms.Compose([transforms.ToTensor()])
    full_dataset = DiffusionDataset(
        diffused_dir="./DMD/120 GRIT/",  # path to your diffused images
        clean_dir="./DMD/Raw/",     # path to your ground truth masks (0 or 1)
        transform=transform
    )

    # We'll pick a small subset (indices 0..4) to overfit
    subset_indices = [0,1,2,3,4]  # or even fewer
    small_subset = Subset(full_dataset, subset_indices)

    # DataLoader with batch_size=1 or 2 for better overfitting
    debug_loader = DataLoader(small_subset, batch_size=1, shuffle=True)

    # Train for 50 epochs or more
    debug_train_model(model, debug_loader, device, num_epochs=50)

    # See if we can get near-perfect predictions
    visualize_predictions(model, debug_loader, device)
