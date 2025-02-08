import os
import re
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
# 3. FIXED DATASET
############################
class DiffusionDataset(Dataset):
    """
    Loads images named like:
      - ./Images/Diffused/Very Diffused/diffused_image0.png
      - ./Images/Raw/raw_image0.png
    etc.
    It parses the index from 'diffused_imageXXX.png' and automatically looks for
    'raw_imageXXX.png' in the Raw folder.
    """
    def __init__(self, diffused_dir, clean_dir, transform=None):
        super().__init__()
        self.diffused_dir = diffused_dir
        self.clean_dir = clean_dir
        self.transform = transform
        
        # We'll gather all filenames that match pattern "diffused_image(\d+).png"
        pattern = re.compile(r'^diffused_image(\d+)\.png$')
        self.diffused_files = []
        
        for fname in os.listdir(self.diffused_dir):
            if pattern.match(fname):
                self.diffused_files.append(fname)
        
        # Sort by numeric index so data order is consistent
        self.diffused_files.sort(key=lambda x: int(pattern.match(x).group(1)))

    def __len__(self):
        return len(self.diffused_files)

    def __getitem__(self, idx):
        # e.g. "diffused_image10.png"
        diffused_filename = self.diffused_files[idx]
        
        # Extract the numeric portion
        match = re.match(r'^diffused_image(\d+)\.png$', diffused_filename)
        if not match:
            raise ValueError(f"File {diffused_filename} does not match 'diffused_image<number>.png' naming.")
        index_str = match.group(1)
        
        # Build the corresponding raw filename: "raw_image10.png"
        raw_filename = f"raw_image{index_str}.png"
        
        # Full paths
        diffused_path = os.path.join(self.diffused_dir, diffused_filename)
        clean_path = os.path.join(self.clean_dir, raw_filename)
        
        # Open images in grayscale
        diffused_img = Image.open(diffused_path).convert('L')
        clean_img = Image.open(clean_path).convert('L')

        # Apply same transforms
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
