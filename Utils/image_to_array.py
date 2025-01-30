import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog



def ConvertSingle():

    # Open a file dialog to select an image
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    file_path = filedialog.askopenfilename(title="Select an Image",
                                        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")])

    if file_path:
        # Load the image using OpenCV in grayscale
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        # Convert to NumPy array
        image_array = np.array(image)

        return image_array
    else:
        return None
    
def LoadImage():
    # Open a file dialog to select an image
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    file_path = filedialog.askopenfilename(title="Select an Image",
                                        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")])

    if file_path:
        
        return file_path
    else:
        return None
    