import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt  
import sys
from HybridResNetUNetModel import ResNetUNet
from DiffusionDataset import DiffusionDataset


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python trainResNetUNet.py <GRIT_VALUE>")
        sys.exit(1)
    
    grit_value = sys.argv[1]
    diffused_dir = f"./DMD/{grit_value} GRIT"
    clean_dir = "./DMD/Raw"
    model_filename = f"resnet_unet_undiffusion_{grit_value}GRIT.pth"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model, loss, optimizer
    model = ResNetUNet(num_classes=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Transforms: e.g. normalizing
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # Create dataset & loader
    dataset = DiffusionDataset(diffused_dir, clean_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    criterion = torch.nn.MSELoss().to(device)
    def train_model(model, dataloader, criterion, optimizer, device, num_epochs=10):
        model.train()
        epoch_losses = []
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, (diffused, clean) in enumerate(dataloader):
                diffused, clean = diffused.to(device), clean.to(device)
                
                outputs = model(diffused)
                loss = criterion(outputs, clean)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            epoch_loss = running_loss / len(dataloader)
            epoch_losses.append(epoch_loss)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        
        print("Training complete!")
        
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, num_epochs+1), epoch_losses, marker='o')
        plt.title('Training Loss vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()
    
    train_model(model, dataloader, criterion,  optimizer, device, num_epochs=10)
    
    # Save model
    torch.save(model.state_dict(), model_filename)
    print(f"Model saved as {model_filename}")



    
