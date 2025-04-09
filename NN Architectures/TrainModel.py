import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
import argparse
import csv
import time
from FourierLoss import FourierLoss
from NPCCLoss import NPCCLoss

# Import your models and dataset classes
from HybridResNetUNetModel import ResNetUNet
from HybridEfficientNetREDNetModel import EfficientNetREDNet
from HybridResNetREDNetModel import ResNetREDNet
from HybridEfficientNetUNetModel import EfficientNetUNet
from DiffusionDataset import DiffusionDataset





# ---------------------------------------------------------------------
# Validate Model (At each epoch)
# ---------------------------------------------------------------------
def validate_model(model, dataloader, device, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    return avg_loss

# ---------------------------------------------------------------------
# Training function 
# ---------------------------------------------------------------------

def train_model(model, loss, train_loader, test_loader, optimizer, device, num_epochs=20, patience=8, output_suffix="", figure_dir="./Figures", csv_dir="./csv"):
    model.train()
    criterion = loss
    train_losses, test_losses, epoch_times = [], [], []
    best_test_loss = float('inf')
    patience_counter = 0
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    for epoch in range(num_epochs):
        t_s = time.time()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        test_loss = validate_model(model, test_loader, device, criterion)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        epoch_times.append(round(time.time() - t_s))

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model_state = model.state_dict()
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Early stopping counter: {patience_counter}/{patience}")

        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(test_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != old_lr:
            print(f"Reducing learning rate: {old_lr:.6f} → {new_lr:.6f}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    os.makedirs(figure_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    plt.figure(figsize=(8, 5))
    epochs_range = range(1, len(train_losses) + 1)
    plt.plot(epochs_range, train_losses, marker='o', label='Train Loss')
    plt.plot(epochs_range, test_losses, marker='o', label='Test Loss')
    plt.title('Training and Test Loss vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    fig_path = f"{figure_dir}/{architecture}_{grit_value}_{cap}_{lossmethod}{output_suffix}.png"
    plt.savefig(fig_path)
    plt.close()

    csv_filename = f"{csv_dir}/{architecture}_{grit_value}_{cap}_{lossmethod}{output_suffix}.csv"
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Epoch', 'Train Loss', 'Test Loss', 'Train Time (s)'])
        for epoch, t_loss, v_loss, e_time in zip(epochs_range, train_losses, test_losses, epoch_times):
            writer.writerow([epoch, t_loss, v_loss, e_time])

    print(f"Performance log saved to {csv_filename}")
    return [best_model_state, best_epoch]

# ---------------------------------------------------------------------
# Main training script
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grit_value", type=str, default="600")
    parser.add_argument("--cap", type=str, default="10000")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batchsz", type=int, default=4)
    parser.add_argument("--architecture", type=str, default="effnet_unet")
    parser.add_argument("--loss", type=str, default="l1")
    parser.add_argument("--greyscale", action="store_true")
    args = parser.parse_args()

    grit_value = args.grit_value
    cap = args.cap
    lr = args.lr
    batchsz = args.batchsz
    architecture = args.architecture
    num_epochs = args.epochs
    lossmethod = args.loss
    greyscale = args.greyscale

    suffix = "_greyscale" if greyscale else ""
    figure_dir = "./FiguresGreyscale" if greyscale else "./Figures"
    csv_dir = "./csvGreyscale" if greyscale else "./csv"

    if greyscale:
        train_diffused_dir = f"./DMD/Greyscale/{grit_value} GRIT"
        train_clean_dir = "./DMD/Greyscale/Raw"
        test_diffused_dir = f"./DMD/Greyscale/Testing/{grit_value} GRIT"
        test_clean_dir = "./DMD/Greyscale/Testing/Raw"
    else:
        train_diffused_dir = f"./DMD/{grit_value} GRIT"
        train_clean_dir = "./DMD/Raw"
        test_diffused_dir = f"./DMD/Testing/{grit_value} GRIT"
        test_clean_dir = "./DMD/Testing/Raw"

    model_filename = f"{architecture}_{grit_value}_{cap}_{lossmethod}{suffix}.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    match lossmethod:
        case "l1": 
            loss = nn.L1Loss()
        case "mse": 
            loss = nn.MSELoss()
        case "Fourier": 
            loss = FourierLoss()
        case "NPCC": 
            loss = NPCCLoss()
        case "Combined":
            loss = CombinedLoss()
        case _: 
            print("No Loss Chosen")
            exit()

    match architecture:
        case "effnet_unet": 
            model = EfficientNetUNet().to(device)
        case "resnet_unet": 
            model = ResNetUNet().to(device)
        case "effnet_rednet": 
            model = EfficientNetREDNet().to(device)
        case "resnet_rednet": 
            model = ResNetREDNet().to(device)
        case _: 
            print("Invalid architecture option")
            exit()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    transform = transforms.Compose([
    transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),  # slight shift & rotation
    transforms.ToTensor()
])

    print("Loading training dataset...")
    train_dataset = DiffusionDataset(train_diffused_dir, train_clean_dir, transform=transform, cap=cap)
    train_loader = DataLoader(train_dataset, batch_size=batchsz, shuffle=True)

    print("Loading testing dataset...")
    test_dataset = DiffusionDataset(test_diffused_dir, test_clean_dir, transform=transform, cap=1000)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print("Beginning training...")
    best_model = train_model(model, loss, train_loader, test_loader, optimizer, device,
                             num_epochs=num_epochs, output_suffix=suffix,
                             figure_dir=figure_dir, csv_dir=csv_dir)

    print(f"Best performing model found at epoch {best_model[1]+1} - Saving the model at this epoch")
    torch.save(best_model[0], model_filename)
    print(f"Model saved as {model_filename}")