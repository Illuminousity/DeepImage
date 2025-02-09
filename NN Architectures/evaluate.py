import torch

import torch.optim as optim
from torch.utils.data import  DataLoader
from torchvision import transforms
from PICNNModel import PICNN, PhysicsLoss, DiffusionDataset
from PIL import Image
import matplotlib.pyplot as plt


# ================================
# TEST / INFERENCE
# ================================
if __name__ == "__main__":
    # 3A: Load your saved model
    model = PICNN()
    checkpoint_path = "picnn_undiffusion_fixed.pth"  # Path to your saved weights
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    # 3B: Define the same image transforms used for training
    test_transform = transforms.Compose([
        transforms.Resize((1024, 768)),   # or whatever size you used
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 3C: Load a diffused test image
    diffused_path = "./Images/Diffused/Very Diffused/diffused_image0.png"  # Path to your diffused image
    diffused_img = Image.open(diffused_path).convert("L")
    actual_path = "./Images/Raw/raw_image0.png"
    actual_img = Image.open(actual_path).convert("L")
    
    # Apply transform & create batch dimension
    input_tensor = test_transform(diffused_img).unsqueeze(0)

    # 3D: Forward pass (inference)
    with torch.no_grad():
        output_tensor = model(input_tensor)

    # 3E: Denormalize the output if you normalized
    output_tensor = output_tensor * 0.5 + 0.5  # invert normalization
    output_tensor = torch.clamp(output_tensor, 0, 1)

    # Convert to PIL for visualization
    output_image = transforms.ToPILImage()(output_tensor.squeeze(0))

    # 3F: Visualize side by side
    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1)
    plt.title("Diffused Input")
    plt.imshow(diffused_img, cmap="gray")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.title("Actual Image")
    plt.imshow(actual_img, cmap="gray")
    plt.axis("off")


    plt.subplot(1,3,3)
    plt.title("PI-CNN Output")
    plt.imshow(output_image, cmap="gray")
    plt.axis("off")
    plt.show()

    # 3G: Optionally save the result
    output_image.save("undiffused_result.png")
    print("Inference complete, saved 'undiffused_result.png'.")