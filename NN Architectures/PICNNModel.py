import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

############################
# 1. PICNN MODEL DEFINITION
############################
class PICNN(nn.Module):
    def __init__(self):
        super(PICNN, self).__init__()
        
        # Encoder
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Decoder
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 1, kernel_size=3, padding=1)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        # Encoder
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        
        # Decoder
        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        x = self.deconv3(x)
        return x

############################
# 2. PHYSICS-BASED LOSS
############################
class PhysicsLoss(nn.Module):
    def __init__(self, lambda_phys=0.1):
        super(PhysicsLoss, self).__init__()
        self.lambda_phys = lambda_phys

        # Define a Laplacian kernel for PDE-based constraints
        laplacian = [[0, 1, 0],
                     [1, -4, 1],
                     [0, 1, 0]]
        self.laplacian_kernel = torch.tensor(laplacian, dtype=torch.float32).view(1, 1, 3, 3)

    def forward(self, pred, target):
        # Basic reconstruction loss (L1)
        recon_loss = F.l1_loss(pred, target)
        
        # Physics-Informed: Laplacian-based regularization
        if pred.is_cuda:
            self.laplacian_kernel = self.laplacian_kernel.cuda()
        pred_lap = F.conv2d(pred, self.laplacian_kernel, padding=1)
        tgt_lap = F.conv2d(target, self.laplacian_kernel, padding=1)
        physics_loss = F.mse_loss(pred_lap, tgt_lap)

        # Combine losses
        total_loss = recon_loss + self.lambda_phys * physics_loss
        return total_loss

############################
# 3. CUSTOM DATASET
############################
class DiffusionDataset(Dataset):
    """
    Expects two folders of identically named images:
      - diffused_dir: e.g. ./Images/Diffused/....
      - clean_dir:    e.g. ./Images/Raw/....
    """
    def __init__(self, diffused_dir, clean_dir, transform=None):
        super().__init__()
        self.diffused_dir = diffused_dir
        self.clean_dir = clean_dir
        self.transform = transform
        
        # List of file names in diffused_dir
        self.diffused_files = sorted(os.listdir(diffused_dir))

    def __len__(self):
        return len(self.diffused_files)

    def __getitem__(self, idx):
        # Build paths
        diffused_filename = self.diffused_files[idx]
        diffused_path = os.path.join(self.diffused_dir, diffused_filename)
        clean_path = os.path.join(self.clean_dir, diffused_filename)
        
        # Open images (grayscale)
        diffused_img = Image.open(diffused_path).convert('L')
        clean_img = Image.open(clean_path).convert('L')

        # Apply same transforms to both
        if self.transform:
            diffused_img = self.transform(diffused_img)
            clean_img = self.transform(clean_img)

        return diffused_img, clean_img

############################
# 4. TRAINING LOOP
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
# EXAMPLE USAGE
############################
if __name__ == '__main__':
    # 1. Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. Create model and push to device
    model = PICNN().to(device)
    
    # 3. Define the loss function and optimizer
    criterion = PhysicsLoss(lambda_phys=0.1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 4. Setup dataset & dataloader
    #    NOTE: We are resizing to 768 (height) x 1024 (width)
    transform = transforms.Compose([
        transforms.Resize((768, 1024)),  # (Height, Width)
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Example normalization
    ])
    
    diffused_dir = "./Images/Diffused/Very Diffused"
    clean_dir = "./Images/Raw"
    train_dataset = DiffusionDataset(diffused_dir, clean_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    
    # 5. Train
    train_model(model, train_loader, criterion, optimizer, device, num_epochs=5)
    
    # 6. Save model
    torch.save(model.state_dict(), "picnn_undiffusion_1024x768.pth")
    print("Model saved.")
