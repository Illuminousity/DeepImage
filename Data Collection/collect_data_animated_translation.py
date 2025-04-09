import numpy as np
import cv2
import time
import sys
import os
import subprocess
from load_emnist import GetImage, LoadDataset
from camera_setup import configure_path

try:
    configure_path()
except ImportError:
    pass

from ALP4 import *
from thorlabs_tsi_sdk.tl_camera import TLCameraSDK

import matplotlib.pyplot as plt
import matplotlib.animation as animation

def FormatImage(num, dmd_placeholder, dataset, invert=False, apply_gradient=False):
    img = GetImage(num, dataset).astype(np.float32)
    img_8bit = img * (2**8 - 1)
    if invert:
        img_8bit = 255 - img_8bit
    img_resized = cv2.resize(img_8bit, (256, 192), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((256, 256), dtype=np.uint8) if dmd_placeholder is None else np.zeros((dmd_placeholder.nSizeY, dmd_placeholder.nSizeX), dtype=np.uint8)
    x_off = (canvas.shape[1] - 256) // 2
    y_off = (canvas.shape[0] - 192) // 2
    canvas[y_off:y_off+192, x_off:x_off+256] = img_resized
    return canvas

def translate_image(image, dx, dy):
    h, w = image.shape[:2]
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)

def process_multiple_captured_images(image_list, binary):
    combined_image = np.median(np.stack(image_list, axis=0), axis=0).astype(np.uint8)
    if binary:
        _, combined_image = cv2.threshold(combined_image, 249, 255, cv2.THRESH_BINARY)
    return combined_image

# Setup
mode = True
emnist = LoadDataset(mode)
num_translations = 60
angle_step = 360 / num_translations
random_indices = [100, 200, 234, 4324, 5842, 1222, 9999, 876, 4312, 60433]
output_folder = "./DMD/Testing/Animated/120 GRIT"
os.makedirs(output_folder, exist_ok=True)

# Compute safe translation radius
canvas_h, canvas_w = 300,300
img_h, img_w = 192, 256
x_off = (canvas_w - img_w) // 2
y_off = (canvas_h - img_h) // 2
max_radius_x = int(x_off * 0.85)
max_radius_y = int(y_off * 0.85)

try:
    with TLCameraSDK() as sdk:
        available_cameras = sdk.discover_available_cameras()
        if len(available_cameras) < 1:
            raise RuntimeError("No cameras detected!")

        with sdk.open_camera(available_cameras[0]) as camera:
            print(f"Opened camera: {camera.model} (SN: {camera.serial_number})")
            camera.roi_width_pixels = 128
            camera.roi_height_pixels = 128
            camera.roi_x_pixels = 728
            camera.roi_y_pixels = 348
            camera.exposure_time_us = 3000    # 2000 for 1500 600 220, 3000 for 120
            camera.frames_per_trigger_zero_for_unlimited = 0
            camera.image_poll_timeout_ms = 1000
            camera.frame_rate_control_value = 200
            camera.is_frame_rate_control_enabled = True
            camera.arm(2)

            dmd = ALP4(version='4.3')
            dmd.Initialize()

            try:
                dmd.SeqAlloc(nbImg=1, bitDepth=1)
                frame_counter = 0
                for img_idx in random_indices:
                    base_img = FormatImage(img_idx, dmd, emnist)
                    for j in range(num_translations):
                        angle = j * angle_step
                        theta = np.deg2rad(angle)
                        dx = int(max_radius_x * np.cos(theta))
                        dy = int(max_radius_y * np.sin(theta))

                        translated = translate_image(base_img, dx, dy)
                        dmd.FreeSeq()
                        dmd.SeqAlloc(nbImg=1, bitDepth=1)
                        dmd.SeqPut(imgData=translated)
                        dmd.Run()
                        time.sleep(0.0075)

                        camera.issue_software_trigger()
                        image_list = []
                        for _ in range(4): # The more frames we use to combine, the more consistent our image becomes
                            # by doing this we prevent scattering artifacts.
                            frame = camera.get_pending_frame_or_null()
                            while frame is None:
                                frame = camera.get_pending_frame_or_null()
                                time.sleep(0.001)
                            image_list.append(np.copy(frame.image_buffer).reshape(
                                camera.image_height_pixels,
                                camera.image_width_pixels
                            ))

                        refined = process_multiple_captured_images(image_list, binary=False)
                        mirrored = cv2.flip(refined, -1)
                        frame_path = os.path.join(output_folder, f"frame_{frame_counter:04d}.png")
                        cv2.imwrite(frame_path, mirrored)
                        frame_counter += 1
                        dmd.Halt()

            finally:
                dmd.FreeSeq()
                camera.disarm()
                cv2.destroyAllWindows()

except Exception as e:
    print(f"DMD/Camera not available, running in simulation mode: {e}")
    import matplotlib.pyplot as plt
    frame_counter = 0
    for img_idx in random_indices:
        base_img = FormatImage(img_idx, None, emnist)
        frames = []
        for j in range(num_translations):
            angle = j * angle_step
            theta = np.deg2rad(angle)
            dx = int(max_radius_x * np.cos(theta))
            dy = int(max_radius_y * np.sin(theta))
            translated = translate_image(base_img, dx, dy)
            frames.append(translated)
            frame_counter += 1

        fig, ax = plt.subplots()
        img_display = ax.imshow(frames[0], cmap='gray')
        ax.axis('off')
        fig.suptitle(f"EMNIST Translating Image Index: {img_idx}")

        def update(frame):
            img_display.set_data(frame)
            return [img_display]

        ani = animation.FuncAnimation(fig, update, frames=frames, interval=40, blit=True, repeat=True)
        plt.show()

# Encode frames into a lossless video using FFmpeg
output_video = "./DMD/Testing/Animated/120 GRIT/120.mkv"
ffmpeg_cmd = [
    "ffmpeg", "-y",
    "-framerate", "10",
    "-i", os.path.join(output_folder, "frame_%04d.png"),
    "-c:v", "ffv1",
    "-preset", "veryslow",
    "-crf", "0",
    output_video
]
subprocess.run(ffmpeg_cmd, check=True)
print("Done capturing and encoding video.")
