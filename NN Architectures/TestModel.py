import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import sys
from HybridResNetUNetModel import ResNetUNetSegmentation
from HybridVGGNetUNetModel import VGGUNetSegmentation
from HybridEfficientNetUNetModel import EfficientNetUNetSegmentation
from DiffusionDataset import DiffusionDataset
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python validate_model.py <GRIT_VALUE> <MODEL_PATH>")
        sys.exit(1)
    
    grit_value = sys.argv[1]
    model_path = sys.argv[2]
    diffused_dir = f"./DMD/Testing/{grit_value} GRIT"
    clean_dir = "./DMD/Testing/Raw"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the model
    model = EfficientNetUNetSegmentation().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Transforms
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # Create dataset & loader
    dataset = DiffusionDataset(diffused_dir, clean_dir,cap=1000, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    criterion = nn.L1Loss()
    total_loss = 0
    
    # Evaluate the model
    with torch.no_grad():
        for i, (diffused, clean) in enumerate(dataloader):
            diffused, clean = diffused.to(device), clean.to(device)
            
            output = model(diffused)
            loss = criterion(output, clean)
            total_loss += loss.item()  # Accumulate the loss
            
            print(f"Batch {i}: Loss = {loss.item():.6f}")
            
            
            # Display a sample prediction
            if i == 1:
                
                output_np = output.squeeze().cpu().numpy()
                pred_probs = torch.sigmoid(output)
                binary_pred = (pred_probs > 0.5).squeeze().cpu().numpy()

                clean_np = clean.squeeze().cpu().numpy()
                diffused_np = diffused.squeeze().cpu().numpy()
                
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                axs[0].imshow(diffused_np, cmap='gray')
                axs[0].set_title("Diffused Input")
                axs[1].imshow(output_np, cmap='gray')
                axs[1].set_title("Net Output")
                axs[2].imshow(clean_np, cmap='gray')
                axs[2].set_title("Ground Truth")
                
                for ax in axs:
                    ax.axis('off')
                plt.show()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Validation Completed. Average L1 Loss: {avg_loss:.6f}")
