import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
import argparse
import csv

from FourierLoss import FourierLoss

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
# Training function with validation at each epoch, loss logging and CSV export.
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# Training function with Early Stopping & Learning Rate Scheduling
# ---------------------------------------------------------------------
def train_model(model, loss, train_loader, test_loader, optimizer, device, num_epochs=20, patience=5):
    model.train()
    criterion = loss
    train_losses = []
    test_losses = []
    
    best_test_loss = float('inf')  # Track the best test loss
    patience_counter = 0  # Track epochs without improvement

    # Implement ReduceLROnPlateau scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Evaluate on test set after each epoch
        test_loss = validate_model(model, test_loader, device, criterion)
        test_losses.append(test_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

        # Early Stopping Logic
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0  # Reset counter if test loss improves
            best_model_state = model.state_dict()  # Store the best model without saving yet
            best_epoch = epoch
        else:
            patience_counter += 1
            print(f"Early stopping counter: {patience_counter}/{patience}")
        
        # Reduce learning rate if test loss plateaus
        old_lr = optimizer.param_groups[0]['lr']  
        scheduler.step(test_loss)
        new_lr = optimizer.param_groups[0]['lr']  

        if new_lr != old_lr:
            print(f"Reducing learning rate: {old_lr:.6f} → {new_lr:.6f}")

        # Stop training if no improvement for 'patience' epochs
        if patience_counter >= patience:
            print("Early stopping triggered. Stopping training.")
            break
    
    # Plot training and test loss curves
    plt.figure(figsize=(8, 5))
    epochs_range = range(1, len(train_losses)+1)
    plt.plot(epochs_range, train_losses, marker='o', label='Train Loss')
    plt.plot(epochs_range, test_losses, marker='o', label='Test Loss')
    plt.title('Training and Test Loss vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"./Figures/{architecture}_{grit_value}_{cap}_{lossmethod}.png")  # Save the figure as a PNG file
    plt.close()  # Close the figure to free up memory
    
    # Write losses to CSV file
    csv_filename = f"./csv/{architecture}_{grit_value}_{cap}_{lossmethod}.csv"
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Epoch', 'Train Loss', 'Test Loss'])
        for epoch, t_loss, v_loss in zip(epochs_range, train_losses, test_losses):
            writer.writerow([epoch, t_loss, v_loss])
    print(f"Performance log saved to {csv_filename}")
    return [best_model_state, best_epoch]

# ---------------------------------------------------------------------
# Main training script
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grit_value", type=str, default="default_grit", help="Grit value for dataset directory")
    parser.add_argument("--cap", type=str, default="default_cap", help="Cap value for dataset")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batchsz", type=int, default=4)
    parser.add_argument("--architecture", type=str, default="effnet_unet", help="Model architecture option")
    parser.add_argument("--loss", type=str, default="l1")
    args = parser.parse_args()
    
    grit_value = args.grit_value
    cap = args.cap
    lr = args.lr
    batchsz = args.batchsz
    architecture = args.architecture
    num_epochs = args.epochs
    lossmethod = args.loss
    # Set up training and testing dataset paths
    train_diffused_dir = f"./DMD/{grit_value} GRIT"
    train_clean_dir = "./DMD/Raw"
    test_diffused_dir = f"./DMD/Testing/{grit_value} GRIT"
    test_clean_dir = "./DMD/Testing/Raw"
    
    model_filename = f"{architecture}_{grit_value}_{cap}_{lossmethod}.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Choose Loss based on argument
    match lossmethod:
        case "l1":
            loss = nn.L1Loss()
        case "mse":
            loss = nn.MSELoss()
        case "Fourier":
            loss = FourierLoss()
        case _:
            print("No Loss Chosen")
            exit()

    # Create model based on architecture choice.
    match architecture:
        case "effnet_unet":
            from HybridEfficientNetUNetModel import EfficientNetUNet
            model = EfficientNetUNet().to(device)
        case "resnet_unet":
            from HybridResNetUNetModel import ResNetUNet
            model = ResNetUNet().to(device)
        case "effnet_rednet":
            from HybridEfficientNetREDNetModel import EfficientNetREDNet
            model = EfficientNetREDNet().to(device)
        case "resnet_rednet":
            from HybridResNetREDNetModel import ResNetREDNet
            model = ResNetREDNet().to(device)
        case _:
            print("Invalid architecture option")
            exit()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)

    transform = transforms.Compose([ 
        transforms.ToTensor(),
    ])
    
    # Create training dataset & dataloader
    print("Loading training dataset...")
    train_dataset = DiffusionDataset(train_diffused_dir, train_clean_dir, transform=transform, cap=cap)
    train_loader = DataLoader(train_dataset, batch_size=batchsz, shuffle=True)
    
    # Create testing dataset & dataloader
    print("Loading testing dataset...")
    test_dataset = DiffusionDataset(test_diffused_dir, test_clean_dir, transform=transform, cap=1000)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    print("Beginning training...")
    best_model = train_model(model,loss, train_loader, test_loader, optimizer, device, num_epochs=num_epochs)
    print("Training finished.")
    
    # Save the trained model.
    print(f"Best performing model found at epoch {best_model[1]+1} - Saving the model at this epoch")
    torch.save(best_model[0], model_filename)
    print(f"Model saved as {model_filename}")
