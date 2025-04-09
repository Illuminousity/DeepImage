import numpy as np
import cv2
import time
import sys
from ALP4 import *
from load_emnist import GetImage, LoadDataset
from thorlabs_tsi_sdk.tl_camera import TLCameraSDK
def grayscale_to_bitplanes(image):
    """ Convert an 8-bit grayscale image to 8 binary bit-planes """
    bit_planes = [(image >> i) & 1 for i in range(8)]  # Extract bit planes
    return bit_planes

def FormatImage(num, DMD, dataset, invert=False, apply_gradient=False):
    img = GetImage(num, dataset).astype(np.float32)
    img_8bit = img * (2**8 - 1)  # Scale to [0..255]

    if invert:
        img_8bit = 255 - img_8bit  # Invert image

    img_resized = cv2.resize(img_8bit, (384, 288), interpolation=cv2.INTER_LINEAR)
    
    if apply_gradient:
        # Create a vertical gradient mask (192 rows, 256 columns)
        gradient = np.linspace(1, 0, 288).reshape(-1, 1)  # Column vector
        gradient = np.tile(gradient, (1, 384))  # Expand to match image width
        img_resized = img_resized * gradient  # Apply gradient
        img_resized = img_resized.astype(np.uint8)  # Convert back to uint8

    canvas = np.zeros((DMD.nSizeY, DMD.nSizeX), dtype=np.uint8)
    x_off = (DMD.nSizeX - 1024) // 2
    y_off = (DMD.nSizeY - 768) // 2
    canvas[y_off:y_off+768, x_off:x_off+1024] = img_resized

    return canvas

def display_bitplanes_on_dmd(bit_planes, dmd,camera):
    """Display 8-bit grayscale image as binary bit-planes using PWM timing."""
    captured_frames = []
    # Stop any running sequence and free previous one
    dmd.Halt()
    time.sleep(0.05)
    try:
        dmd.FreeSeq()
    except Exception as e:
        print(f"Warning: FreeSeq failed - {e}")
    
    # Allocate new sequence (bitDepth=1 for binary)
    dmd.SeqAlloc(nbImg=1, bitDepth=1)
    
    # Ensure bit-planes are in the correct format
    binary_images = [(bit_plane * 255).astype(np.uint8) for bit_plane in bit_planes]
    for i, binary_image in enumerate(binary_images):
        flattened_image = binary_image.flatten().astype(np.uint8)
        
        # Debug print before sending
        
        # Send the single bit-plane to the DMD
        dmd.SeqPut(imgData=flattened_image)
        dmd.SetTiming(pictureTime=(2 ** i)*75)  # Set timing for current bit-plane
        dmd.Run()
        
               # Capture frame from the camera during display
        camera.issue_software_trigger()
        frame = camera.get_pending_frame_or_null()
        while frame is None:
            frame = camera.get_pending_frame_or_null()
            time.sleep(0.001)
        captured_frames.append(np.copy(frame.image_buffer).reshape(
            camera.image_height_pixels,
            camera.image_width_pixels
        ))
        
        # Allow time for exposure before moving to the next bit-plane
        time.sleep(0.01)
        dmd.Halt()
    
    
    return captured_frames
    
    



try:
    from camera_setup import configure_path
    configure_path()
except ImportError:
    configure_path = None
# Load dataset
mode = True
emnist = LoadDataset(mode)

dmd = ALP4(version='4.3')
dmd.Initialize()

# Initialize Camera
with TLCameraSDK() as sdk:
    available_cameras = sdk.discover_available_cameras()
    if len(available_cameras) < 1:
        print("No cameras detected!")
        sys.exit()

    with sdk.open_camera(available_cameras[0]) as camera:
        print(f"Opened camera: {camera.model} (SN: {camera.serial_number})")
        camera.roi_width_pixels = 1024
        camera.roi_height_pixels = 1024
        camera.exposure_time_us = 153  # Adjust exposure to capture all bit-planes
        camera.frames_per_trigger_zero_for_unlimited = 0 # Avoid buffer error
        camera.image_poll_timeout_ms = 0
        camera.frame_rate_control_value = 6.53

        camera.arm(2)
        
        try:
            for i in range(1000):  # Capture 100 images for testing
                grayscale_image = FormatImage(i, dmd,emnist,False,True)
                bit_planes = grayscale_to_bitplanes(grayscale_image)
                
                captured_frames = display_bitplanes_on_dmd(bit_planes, dmd,camera)
                time.sleep(0.01)  # Short delay to sync camera
                
                
                for j, frame in enumerate(captured_frames):
                    cv2.imwrite(f"./DMD/Greyscale/Raw/captured_frame_{i}_bitplane_{j}.png", frame)
                
                #dmd.Halt()
        except KeyboardInterrupt:
            print("Interrupted by user.")
        finally:
            dmd.Halt()
            camera.disarm()

print("Program completed.")
