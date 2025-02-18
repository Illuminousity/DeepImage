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
from DiffusionDataset import DiffusionDataset

def train_model(model, dataloader, optimizer, device, num_epochs=10):
    model.train()
    criterion = nn.BCEWithLogitsLoss()  # or nn.MSELoss()

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
    if len(sys.argv) != 2:
        print("Usage: python trainResNetUNet.py <GRIT_VALUE>")
        sys.exit(1)

    grit_value = sys.argv[1]
    diffused_dir = f"./DMD/{grit_value} GRIT"
    clean_dir = "./DMD/Raw"
    model_filename = f"resnet_unet_simplified_{grit_value}.pth"
    model2_filename = f"vggnet_unet_simplified_{grit_value}.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model
    model = ResNetUNetSegmentation().to(device)
    model2 = VGGUNetSegmentation().to(device)
    # Optimizer (Adam or SGD)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    optimizer2 = optim.Adam(model2.parameters(), lr=1e-3)
    # Simple transforms
    transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),  # small rotation
    transforms.ToTensor()
])


    # Create dataset & loader
    dataset = DiffusionDataset(diffused_dir, clean_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Train
    train_model(model, dataloader, optimizer, device, num_epochs=15)
    train_model(model2, dataloader, optimizer2, device, num_epochs=15)

    # Save model
    torch.save(model.state_dict(), model_filename)
    torch.save(model2.state_dict(), model2_filename)
    print(f"Model saved as {model_filename}")
