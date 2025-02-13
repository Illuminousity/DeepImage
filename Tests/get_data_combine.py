import numpy as np
import cv2
import time
import sys
from ALP4 import *
from load_emnist import GetImage
from thorlabs_tsi_sdk.tl_camera import TLCameraSDK, OPERATION_MODE
import matplotlib.pyplot as plt 

# Function to format an image for the DMD with inversion
def FormatImage(num, DMD, invert=False):
    img = GetImage(num).astype(np.float32)
    img_8bit = img * (2**8 - 1)  # Scale to [0..255]
    if invert:
        img_8bit = 255 - img_8bit  # Invert image
    img_resized = cv2.resize(img_8bit, (256, 192), interpolation=cv2.INTER_LINEAR)
    img_resized = img_resized.astype(np.uint8)
    canvas = np.zeros((DMD.nSizeY, DMD.nSizeX), dtype=np.uint8)
    x_off = (DMD.nSizeX - 256) // 2
    y_off = (DMD.nSizeY - 192) // 2
    canvas[y_off:y_off+192, x_off:x_off+256] = img_resized
    return canvas

# Function to process captured images
def process_captured_images(original_img, inverted_img):
    # Binarize both images
    # 127 - RAW
    # 10 - Diffused (More sensitive to data)
    _, binary_orig = cv2.threshold(original_img, 127, 255, cv2.THRESH_BINARY)
    _, binary_inv = cv2.threshold(inverted_img, 255, 255, cv2.THRESH_BINARY)
    
    
 

    

    

    # Re-invert the inverted image
    inverted_img = 255 - inverted_img
    
    # Perform AND operation
    refined_img = np.bitwise_and(original_img, inverted_img) * 4
    _, bin_refined = cv2.threshold(refined_img, 0.01, 255, cv2.THRESH_BINARY)
    return refined_img

try:
    from camera_setup import configure_path
    configure_path()
except ImportError:
    configure_path = None

# Initialize the DMD
dmd = ALP4(version='4.3')
dmd.Initialize()
dmd.SeqAlloc(nbImg=1, bitDepth=8)
dmd.SetTiming(pictureTime=5000)

# Initialize the camera
with TLCameraSDK() as sdk:
    available_cameras = sdk.discover_available_cameras()
    if len(available_cameras) < 1:
        print("No cameras detected!")
        sys.exit()

    with sdk.open_camera(available_cameras[0]) as camera:
        print(f"Opened camera: {camera.model} (SN: {camera.serial_number})")
        camera.roi_width_pixels = 128
        camera.roi_height_pixels = 128
        camera.roi_x_pixels = 728
        camera.roi_y_pixels = 348
        camera.exposure_time_us = 2000
        camera.frames_per_trigger_zero_for_unlimited = 0
        camera.image_poll_timeout_ms = 0
        camera.frame_rate_control_value = 200
        camera.is_frame_rate_control_enabled = True
        camera.arm(2)

        try:
            for i in range(0, 999):
                # Capture original image
                image = FormatImage(i, dmd, invert=False)
                dmd.SeqPut(imgData=image)
                dmd.Run()
                time.sleep(0.05)

                camera.issue_software_trigger()
                frame = camera.get_pending_frame_or_null()
                while frame is None:
                    frame = camera.get_pending_frame_or_null()
                    time.sleep(0.001)
                original_capture = np.copy(frame.image_buffer).reshape(
                    camera.image_height_pixels,
                    camera.image_width_pixels
                )
                dmd.Halt()
                # Capture inverted image
                image_inv = FormatImage(i, dmd, invert=True)
                dmd.SeqPut(imgData=image_inv)
                dmd.Run()
                time.sleep(0.05)

                camera.issue_software_trigger()
                frame = camera.get_pending_frame_or_null()
                while frame is None:
                    frame = camera.get_pending_frame_or_null()
                    time.sleep(0.001)
                inverted_capture = np.copy(frame.image_buffer).reshape(
                    camera.image_height_pixels,
                    camera.image_width_pixels
                )
                
                # Process images to refine
                refined_image = process_captured_images(original_capture, inverted_capture)
                nd_image_array = np.dstack([refined_image]*3).astype(np.uint8)
                # Mirror the image both horizontally and vertically
                mirrored_image = cv2.flip(nd_image_array, -1)
                
                filename = f"./DMD/120 GRIT/captured_frame_{i}.png"



                cv2.imwrite(filename, mirrored_image)
                print(f"Saved {filename}")

                dmd.Halt()
        except KeyboardInterrupt:
            print("Interrupted by user.")
        finally:
            dmd.FreeSeq()
            dmd.Free()
            cv2.destroyAllWindows()
            camera.disarm()

print("Program completed.")
