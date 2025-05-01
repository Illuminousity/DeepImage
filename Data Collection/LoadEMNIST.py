import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def LoadDataset(mode):

    transform = transforms.Compose([transforms.ToTensor()])

    # Load the EMNIST dataset (letters subset)
    return torchvision.datasets.EMNIST(root="./data", split="letters", train=mode, download=True, transform=transform)

def GetImage(index,dataset=LoadDataset(True)):
    # Define transform to convert PIL images to tensors
    
    emnist = dataset

    # Example: Load a single image
    image, label = emnist[index]  # Get first image and label


    # Convert to NumPy array for processing
    import numpy as np
    emnist_image = image.numpy().squeeze()  # Convert tensor to NumPy
    return emnist_image


if __name__ == "__main__":

    emnist = LoadDataset(1)

    fig, ax = plt.subplots(1, 1, figsize=(3,4))
    image, label = emnist[0]
    ax.imshow(image.numpy().squeeze(),cmap="gray")
    ax.axis('off')
    print(f"Image {0}/{len(emnist)} loaded")


