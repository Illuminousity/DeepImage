import numpy as np
from ALP4 import *
import time
from load_emnist import GetImage
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import numpy as np



def FormatImage(num,DMD):

    return cv2.resize(GetImage(num), (DMD.nSizeX, DMD.nSizeY), interpolation=cv2.INTER_LINEAR) * (2**8-1)



# Load the Vialux .dll
DMD = ALP4(version = '4.3')
# Initialize the device
DMD.Initialize()
# Binary amplitude image (0 or 1)
bitDepth = 8
imgSeq=[]

image1 = FormatImage(42,DMD) 
print(image1.__class__)
image2 = FormatImage(2,DMD)



imgSeqTest3 = np.ravel(np.array(np.resize(GetImage(15),[1024,768]))*(2**8-1))
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
DMD.SetTiming(pictureTime = 62500)

# Run the sequence in an infinite loop
DMD.Run()

time.sleep(999)

# Stop the sequence display
DMD.Halt()
# Free the sequence from the onboard memory
DMD.FreeSeq()
# De-allocate the device
DMD.Free()