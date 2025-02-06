import torch
import matplotlib.pyplot as plt
from DataFeeder import DiffusionDataset, DataLoader
def evaluate(model, test_loader):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for diffused_images, clean_images in test_loader:
            diffused_images, clean_images = diffused_images.to(device), clean_images.to(device)

            # Forward pass
            outputs = model(diffused_images)

            # Convert tensors to images for visualization
            diffused_img = diffused_images[0].cpu().squeeze().numpy()
            clean_img = clean_images[0].cpu().squeeze().numpy()
            output_img = outputs[0].cpu().squeeze().numpy()

            # Plot images
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(diffused_img, cmap='gray')
            axs[0].set_title("Diffused Image")
            axs[1].imshow(clean_img, cmap='gray')
            axs[1].set_title("Ground Truth")
            axs[2].imshow(output_img, cmap='gray')
            axs[2].set_title("Undiffused Output")
            plt.show()

            break  # Show only one example

# Create test dataset loader
test_dataset = DiffusionDataset(diffused_dir="data/test_diffused", clean_dir="data/test_clean", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Run evaluation
evaluate(model, test_loader)
