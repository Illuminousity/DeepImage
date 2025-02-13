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

def FormatImage(num, DMD):
    """
    Retrieve an EMNIST image, resize it to 256x192, then center it 
    onto a 1024x768 canvas. Ensures pixel values are scaled to [0..255].
    """
    # 1. Get the EMNIST image as a float array in [0..1], presumably 28x28 or similar
    img = GetImage(num).astype(np.float32)
    
    # 2. Scale the pixel values from [0..1] to [0..255]
    img_8bit = img * (2**8 - 1)  # same as *255
    
    # 3. Resize that to 256x192
    #    (Remember cv2.resize expects (width, height))
    img_resized = cv2.resize(img_8bit, (256, 192), interpolation=cv2.INTER_LINEAR)
    
    # 4. Convert to 8-bit integer
    img_resized = img_resized.astype(np.uint8)
    
    # 5. Create a black 1024x768 canvas
    #    shape => (height=768, width=1024)
    canvas = np.zeros((DMD.nSizeY, DMD.nSizeX), dtype=np.uint8)
    
    # 6. Compute the offsets to center the 256x192 block
    x_off = (DMD.nSizeX - 256) // 2
    y_off = (DMD.nSizeY - 192) // 2
    
    # 7. Place the resized EMNIST image onto the canvas
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
image1 = FormatImage(3,DMD) 
imgSeqTest3 = np.ravel(np.array(np.resize(GetImage(105),[1024,768]))*(2**8-1))
for i in range(1,20):
    image= GetImage(i)
    if i == 1:
        imgSeq = (np.ravel(np.array(np.resize(image,[1024,768]))))
    else:
        imgSeq = np.concatenate([imgSeq,(np.ravel(np.array(np.resize(image,[1024,768]))))])

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
DMD.SetTiming(pictureTime = 5000)

# Run the sequence in an infinite loop
DMD.Run()

time.sleep(999)

# Stop the sequence display
DMD.Halt()
# Free the sequence from the onboard memory
DMD.FreeSeq()
# De-allocate the device
DMD.Free()