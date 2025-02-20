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

    emnist = LoadDataset()

    fig, ax = plt.subplots(1, 1, figsize=(1,1))
    for i in range():
        image, label = emnist[i]
        ax[i].imshow(image,cmap="gray")
        ax[i].axis('off')
        print(f"Image {i}/{len(emnist)} loaded")


