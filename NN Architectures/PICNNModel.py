import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms

# Define the Physics-Informed CNN (PI-CNN) for Image Undiffusion
class PICNN(nn.Module):
    def __init__(self):
        super(PICNN, self).__init__()
        
        # Encoder: Feature extraction with convolution layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # First layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # Decoder: Reconstruction with transpose convolutions
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1)
        
        # Activation function
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        x = self.deconv3(x)  # Output layer
        return x

# Define the physics-based loss function
class PhysicsLoss(nn.Module):
    def __init__(self, lambda_phys=0.1):
        super(PhysicsLoss, self).__init__()
        self.lambda_phys = lambda_phys
    
    def forward(self, pred, target):
        # Reconstruction loss (L1 loss)
        recon_loss = F.l1_loss(pred, target)
        
        # Physics-informed regularization term: Enforcing inverse diffusion constraints
        # Compute Laplacian to estimate diffusion effects
        laplacian_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3).to(pred.device)
        pred_laplace = F.conv2d(pred, laplacian_kernel, padding=1)
        target_laplace = F.conv2d(target, laplacian_kernel, padding=1)
        physics_loss = F.mse_loss(pred_laplace, target_laplace)
        
        # Total loss
        total_loss = recon_loss + self.lambda_phys * physics_loss
        return total_loss

# Initialize model, loss function, and optimizer
model = PICNN()
loss_function = PhysicsLoss(lambda_phys=0.1)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Example of how to use the model
if __name__ == "__main__":
    # Create a dummy input (diffused image) and target (clean image)
    diffused_image = torch.randn(1, 1, 1024, 768)  # Simulated input
    clean_image = torch.randn(1, 1, 1024, 768)  # Ground truth
    
    # Forward pass
    output = model(diffused_image)
    
    # Compute loss
    loss = loss_function(output, clean_image)
    print("Loss:", loss.item())
    
    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
