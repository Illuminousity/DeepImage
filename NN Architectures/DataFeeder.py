from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class DiffusionDataset(Dataset):
    def __init__(self, diffused_dir, clean_dir, transform=None):
        self.diffused_dir = diffused_dir
        self.clean_dir = clean_dir
        self.transform = transform
        self.image_names = sorted(os.listdir(diffused_dir))  # Ensure alignment between inputs and targets

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        diffused_path = os.path.join(self.diffused_dir, self.image_names[idx])
        clean_path = os.path.join(self.clean_dir, self.image_names[idx])  # Assuming paired data

        # Load images
        diffused_image = Image.open(diffused_path).convert('L')  # Convert to grayscale if needed
        clean_image = Image.open(clean_path).convert('L')

        # Apply transformations
        if self.transform:
            diffused_image = self.transform(diffused_image)
            clean_image = self.transform(clean_image)

        return diffused_image, clean_image

# Define Image Transformations
transform = transforms.Compose([
    transforms.Resize((1024, 768)),
    transforms.ToTensor(),  # Convert image to PyTorch Tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize between -1 and 1
])

# Create Dataset and DataLoader
train_dataset = DiffusionDataset(diffused_dir="../Images/Diffused", clean_dir="../Images/Raw", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
