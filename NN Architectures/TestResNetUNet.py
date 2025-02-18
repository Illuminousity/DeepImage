import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import sys
from HybridResNetUNetModel import ResNetUNetSegmentation
from HybridVGGNetUNetModel import VGGUNetSegmentation
from DiffusionDataset import DiffusionDataset
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python validate_model.py <GRIT_VALUE> <MODEL_PATH> <MODEL_PATH 2>")
        sys.exit(1)
    
    grit_value = sys.argv[1]
    model_path = sys.argv[2]
    model2_path = sys.argv[3]
    diffused_dir = f"./DMD/Testing Data/{grit_value} GRIT"
    clean_dir = "./DMD/Testing Data/RAW"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the model
    model = ResNetUNetSegmentation().to(device)
    model2 = VGGUNetSegmentation().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model2.load_state_dict(torch.load(model2_path, map_location=device))
    model2.eval()
    
    # Transforms
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # Create dataset & loader
    dataset = DiffusionDataset(diffused_dir, clean_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    
    # Evaluate the model
    with torch.no_grad():
        for i, (diffused, clean) in enumerate(dataloader):
            diffused, clean = diffused.to(device), clean.to(device)
            
            total_loss = 1
            
            output = model(diffused)
            output2 = model(diffused)
            
            # Display a sample prediction
            if i == 10:
                
                output_np = output.squeeze().cpu().numpy()
                output2_np = output2.squeeze().cpu().numpy()
                pred_probs = torch.sigmoid(output)
                binary_pred = (pred_probs > 0.5).squeeze().cpu().numpy()

                clean_np = clean.squeeze().cpu().numpy()
                diffused_np = diffused.squeeze().cpu().numpy()
                
                fig, axs = plt.subplots(1, 4, figsize=(16, 4))
                axs[0].imshow(diffused_np, cmap='gray')
                axs[0].set_title("Diffused Input")
                axs[1].imshow(output_np, cmap='gray')
                axs[1].set_title("ResNetUNet Output")
                axs[2].imshow(output2_np, cmap="gray")
                axs[2].set_title("VGGNetUNet Output")
                axs[3].imshow(clean_np, cmap='gray')
                axs[3].set_title("Ground Truth")
                
                for ax in axs:
                    ax.axis('off')
                plt.show()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Validation Completed. Average MSE Loss: {avg_loss:.6f}")
