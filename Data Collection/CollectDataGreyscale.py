import numpy as np
import cv2
import time
import sys
from ALP4 import *
from LoadEMNIST import GetImage, LoadDataset
from thorlabs_tsi_sdk.tl_camera import TLCameraSDK, OPERATION_MODE

# Function to format an image for the DMD with inversion
def FormatImage(num, DMD, dataset, invert=False, apply_gradient=False):
    img = GetImage(num, dataset).astype(np.float32)
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

# Function to process multiple captured images
def process_multiple_captured_images(image_list):
    # Compute the pixel-wise median to retain consistent features
    combined_image = np.median(np.stack(image_list, axis=0), axis=0).astype(np.uint8)
    return combined_image

try:
    from camera_setup import configure_path
    configure_path()
except ImportError:
    configure_path = None
# mode = False means Test Data, mode = True means Training Data
time_s = time.time()
mode = True
emnist = LoadDataset(mode)

# Initialize the DMD
dmd = ALP4(version='4.3')
dmd.Initialize()
dmd.SeqAlloc(nbImg=1, bitDepth=8)
dmd.SetTiming(pictureTime=4000)

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
        camera.exposure_time_us = 2000 # 70 for RAW, 1000 for 1500 GRIT & 600 GRIT, 1200 for 220 GRIT & 120 GRIT
        camera.frames_per_trigger_zero_for_unlimited = 0
        camera.image_poll_timeout_ms = 1000
        camera.frame_rate_control_value = 125
        camera.is_frame_rate_control_enabled = True
        camera.arm(2)

        try:
            for i in range(0, 60000):
                
                dmd.FreeSeq()
                dmd.SeqAlloc(nbImg=1, bitDepth=8)
                image = FormatImage(i, dmd, emnist,invert=False,apply_gradient=True)
                dmd.SeqPut(imgData=image)
                dmd.Run()
                time.sleep(0.0075)

                image_stack = []
                for _ in range(4):
                    camera.issue_software_trigger()
                    frame = camera.get_pending_frame_or_null()
                    while frame is None:
                        frame = camera.get_pending_frame_or_null()
                        time.sleep(0.001)
                    
                    image_np = np.copy(frame.image_buffer).reshape(
                        camera.image_height_pixels,
                        camera.image_width_pixels
                    )
                    image_stack.append(image_np)

                # Sum the 10 frames and normalize to 8-bit
                summed_image = np.sum(image_stack, axis=0)
                summed_image = np.clip(summed_image, 0, 255).astype(np.uint8)

                # Mirror the image
                mirrored_image = cv2.flip(summed_image, -1)

                filename = f"./DMD/Greyscale/120 GRIT/captured_frame_{i}.png"
                cv2.imwrite(filename, mirrored_image)

                dmd.Halt()
        except KeyboardInterrupt:
            print("Interrupted by user.")
        finally:
            dmd.FreeSeq()
            dmd.Free()
            cv2.destroyAllWindows()
            camera.disarm()

print("Program completed.")
print(f"Time Taken: {time.time()-time_s}")
