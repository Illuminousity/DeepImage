import numpy as np
from ALP4 import *
import time
from load_emnist import GetImage
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2

import numpy as np
import cv2

def FormatImage(num, DMD,invert=False, apply_gradient=False):
    img = GetImage(num).astype(np.float32)
    img_8bit = img * (2**8 - 1)  # Scale to [0..255]

    if invert:
        img_8bit = 255 - img_8bit  # Invert image

    img_resized = cv2.resize(img_8bit, (256, 192), interpolation=cv2.INTER_LINEAR)
    
    if apply_gradient:
        # Create a vertical gradient mask (192 rows, 256 columns)
        gradient = np.linspace(1, 0, 192).reshape(-1, 1)  # Column vector
        gradient = np.tile(gradient, (1, 256))  # Expand to match image width
        img_resized = img_resized * gradient  # Apply gradient
        img_resized = img_resized.astype(np.uint8)  # Convert back to uint8

    canvas = np.zeros((DMD.nSizeY, DMD.nSizeX), dtype=np.uint8)
    x_off = (DMD.nSizeX - 256) // 2
    y_off = (DMD.nSizeY - 192) // 2
    canvas[y_off:y_off+192, x_off:x_off+256] = img_resized

    return canvas


# Load the Vialux .dll
DMD = ALP4(version = '4.3')
# Initialize the device
DMD.Initialize()
# Binary amplitude image (0 or 1)
bitDepth = 8
imgSeq=[]

image1 = FormatImage(3,DMD) 
print(image1.__class__)
image2 = FormatImage(2,DMD)
image1 = np.ones((DMD.nSizeY, DMD.nSizeX), dtype=np.uint8)*255
image1 = FormatImage(3,DMD,False,False)


                
fig, axs = plt.subplots(1, 1, figsize=(12, 4))
axs.imshow(image1,cmap='gray')
axs.set_title("Gradient")

                
plt.show()
imgSeqTest3 = np.ravel(np.array(np.resize(GetImage(105),[1024,384]))*(2**8-1))
for i in range(1,20):
    image= GetImage(i)
    if i == 1:
        imgSeq = (np.ravel(np.array(np.resize(image,[512,384]))))
    else:
        imgSeq = np.concatenate([imgSeq,(np.ravel(np.array(np.resize(image,[1024,384]))))])

#imgWhite = np.ones([DMD.nSizeY,DMD.nSizeX])*(2**8-1)

#imgBlack = np.zeros([DMD.nSizeY,DMD.nSizeX])
#imgSeqTest2  = np.concatenate([imgSeqTest3,imgSeqTest])
#image = transforms.ToPILImage()(imgSeqTest)
#image.save("your_file2.png")
# Allocate the onboard memory for the image sequence
DMD.SeqAlloc(nbImg = 1, bitDepth = 8)
# Send the image sequence as a 1D list/array/numpy array
DMD.SeqPut(imgData = image1)
# Set image rate to 50 Hz
DMD.SetTiming(pictureTime = 4000)

# Run the sequence in an infinite loop
DMD.Run()

time.sleep(999)

# Stop the sequence display
DMD.Halt()
# Free the sequence from the onboard memory
DMD.FreeSeq()
# De-allocate the device
DMD.Free()