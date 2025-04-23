import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from PIL import Image
import os
import sys
from math import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../NNArchitectures/')))
from HybridEfficientNetUNetModel import EfficientNetUNet
from HybridResNetUNetModel import ResNetUNet

def plot_feature_maps(tensor, layer_name, num_channels=36):
    num_channels = min(num_channels, tensor.shape[1])
    if num_channels == 1:
        fig, ax = plt.subplots(figsize=(4, 4))
        fmap = tensor[0, 0].detach().cpu().numpy()
        ax.imshow(fmap, cmap='viridis')
        ax.axis('off')
    else:
        grid_size = ceil(sqrt(num_channels))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size, grid_size))
        axes = axes.flat
        for i in range(num_channels):
            fmap = tensor[0, i].detach().cpu().numpy()
            ax = next(axes)
            ax.imshow(fmap, cmap='viridis')
            ax.axis('off')
        # Hide remaining unused plots
        for ax in axes:
            ax.axis('off')
    plt.suptitle(layer_name)
    plt.tight_layout()
    plt.show()

def visualize_filters(layer_weights, layer_name, num_filters=6):
    num_filters = min(num_filters, layer_weights.shape[0])
    fig, axes = plt.subplots(1, num_filters, figsize=(num_filters, 2))
    for i in range(num_filters):
        filt = layer_weights[i, 0].detach().cpu().numpy()  
        axes[i].imshow(filt, cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'Filter {i}')
    plt.suptitle(f"{layer_name} Filters")
    plt.tight_layout()
    plt.show()

def extract_and_visualize_filters(model, max_filters=6):
    print("\nVisualizing filters from encoder conv layers...\n")
    for name, module in model.encoder.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            try:
                weights = module.weight
                visualize_filters(weights, f"{name}", num_filters=max_filters)
            except Exception as e:
                print(f"Could not visualize {name}: {e}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetUNet().to(device)
    model.eval()

    # Load your image
    image_path = "./DMD/Greyscale/1500 Grit/captured_frame_0.png"
    img = Image.open(image_path).convert('L')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        # Encoder outputs (feature maps)
        s1, s2, s3, s4, s5 = model.encoder(img_tensor)

        # Visualize feature maps
        plot_feature_maps(img_tensor, "Input Image", num_channels=1)
        plot_feature_maps(s1, "Feature Maps")
        plot_feature_maps(s2, "Feature Maps")
        plot_feature_maps(s3, "Feature Maps")
        plot_feature_maps(s4, "Feature Maps")
        plot_feature_maps(s5, "Feature Maps")

        # Visualize reconstructed output
        output = model(img_tensor)
        plot_feature_maps(output, "Decoder - Reconstructed Output", num_channels=1)

        # Visualize conv filters
        extract_and_visualize_filters(model)

if __name__ == "__main__":
    main()
