import torch
import torch.optim as optim
from torchvision.transforms import transforms
from PICNNModel import PICNN, PhysicsLoss
from DataFeeder import DiffusionDataset, DataLoader

# Initialize model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PICNN().to(device)
loss_function = PhysicsLoss(lambda_phys=0.1).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
def train(model, dataloader, loss_function, optimizer, num_epochs=10):
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0.0

        for diffused_images, clean_images in dataloader:
            diffused_images, clean_images = diffused_images.to(device), clean_images.to(device)

            # Forward pass
            outputs = model(diffused_images)
            loss = loss_function(outputs, clean_images)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")


# Define Image Transformations
transform = transforms.Compose([
    transforms.Resize((1024, 768)),
    transforms.ToTensor(),  # Convert image to PyTorch Tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize between -1 and 1
])

# Create Dataset and DataLoader
train_dataset = DiffusionDataset(diffused_dir="../Images/Diffused/Very Diffused", clean_dir="../Images/Raw", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Run Training
train(model, train_loader, loss_function, optimizer, num_epochs=10)
