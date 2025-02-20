import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

import matplotlib.pyplot as plt
import sys
import os

from HybridResNetUNetModel import ResNetUNetSegmentation
from HybridVGGNetUNetModel import VGGUNetSegmentation
from HybridEfficientNetUNetModel import EfficientNetUNetSegmentation
from DiffusionDataset import DiffusionDataset

def train_model(model, dataloader, optimizer, device, num_epochs=10):
    model.train()
    criterion = nn.L1Loss()  # or nn.MSELoss()

    epoch_losses = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (diffused, clean) in enumerate(dataloader):
            diffused, clean = diffused.to(device), clean.to(device)

            # Forward
            outputs = model(diffused)

            # Standard pixel-level loss
            loss = criterion(outputs, clean)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        epoch_losses.append(epoch_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # Plot the final training curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_epochs+1), epoch_losses, marker='o')
    plt.title('Training Loss vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python trainResNetUNet.py <GRIT_VALUE>")
        sys.exit(1)
    cap = sys.argv[2]
    grit_value = sys.argv[1]
    diffused_dir = f"./DMD/{grit_value} GRIT"
    clean_dir = "./DMD/Raw"
    model_filename = f"effnet_unet_simplified_{grit_value}_{cap}size.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model
    model = EfficientNetUNetSegmentation().to(device)
    # Optimizer (Adam or SGD)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # Simple transforms
    transform = transforms.Compose([ 
    transforms.ToTensor()
])


    # Create dataset & loader
    dataset = DiffusionDataset(diffused_dir, clean_dir, transform=transform,cap=cap)

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Train
    train_model(model, dataloader, optimizer, device, num_epochs=15)

    # Save model
    torch.save(model.state_dict(), model_filename)
    print(f"Model saved as {model_filename}")
