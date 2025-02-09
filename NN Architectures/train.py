
import torch

import torch.optim as optim
from torch.utils.data import  DataLoader
from torchvision import transforms
from PICNNModel import PICNN, PhysicsLoss, DiffusionDataset


############################
#TRAINING LOOP
############################
def train_model(model, dataloader, criterion, optimizer, device, num_epochs=5):
    model.train()  # Set to training mode
    
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
        
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    
    print("Training complete!")

############################
# USAGE EXAMPLE
############################
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model, loss, optimizer
    model = PICNN().to(device)
    criterion = PhysicsLoss(lambda_phys=0.1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Transforms: e.g. resizing to 256x256 & normalizing
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Folders
    diffused_dir = "./Images/Diffused/Very Diffused"
    clean_dir = "./Images/Raw"
    
    # Create dataset & loader
    dataset = DiffusionDataset(diffused_dir, clean_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Train
    train_model(model, dataloader, criterion, optimizer, device, num_epochs=5)
    
    # Save model
    torch.save(model.state_dict(), "picnn_undiffusion_fixed.pth")
    print("Model saved.")
