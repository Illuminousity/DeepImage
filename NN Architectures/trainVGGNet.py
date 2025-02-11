import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt  # <-- ADDED: For plotting

from VGGNetModel import VGGNet20, HelmholtzLoss, DiffusionDataset
import math
############################
# TRAINING LOOP WITH PLOTTING
############################
def train_model(model, dataloader, criterion, optimizer, device, num_epochs=5):
    model.train()  # Set to training mode
    
    epoch_losses = []  # <-- ADDED: List to store loss each epoch
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        for i, (diffused, clean) in enumerate(dataloader):
            diffused, clean = diffused.to(device), clean.to(device)
            
            # Forward pass
            outputs = model(diffused)
            loss = criterion(outputs, clean)
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Compute average loss for this epoch
        epoch_loss = running_loss / len(dataloader)
        epoch_losses.append(epoch_loss)  # <-- ADDED: Store it
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    
    print("Training complete!")
    
    # ============================
    # PLOT THE EPOCH-LOSS CURVE
    # ============================
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_epochs+1), epoch_losses, marker='o')
    plt.title('Training Loss vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

############################
# USAGE EXAMPLE
############################
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model, loss, optimizer
    model = VGGNet20().to(device)
    criterion = HelmholtzLoss(wave_number=(2*3.1415926 / 660**-9), lambda_phys=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Transforms: e.g.  normalizing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Folders
    diffused_dir = "./Images/Diffused/50 Diffused"
    clean_dir = "./Images/Raw"
    
    # Create dataset & loader
    dataset = DiffusionDataset(diffused_dir, clean_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Train & plot
    train_model(model, dataloader, criterion, optimizer, device, num_epochs=10)
    
    # Save model
    torch.save(model.state_dict(), "vggnet20_undiffusion_10_full_physics.pth")
    print("Model saved.")
