import cv2
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from torchvision import transforms
# Import your models and dataset classes
from HybridResNetUNetModel import ResNetUNet
from HybridEfficientNetREDNetModel import EfficientNetREDNet
from HybridResNetREDNetModel import ResNetREDNet
from HybridEfficientNetUNetModel import EfficientNetUNet
from DiffusionDataset import DiffusionDataset

# --- Settings ---
VIDEO_PATH = './DMD/Testing/Animated/220 GRIT/220.mkv'
OUTPUT_VIDEO_PATH = './DMD/Testing/Animated/220 GRIT/effnet_rednet_reconstructed_output.mp4'
FPS_DISPLAY = True

# --- Load trained model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EfficientNetREDNet()
model.load_state_dict(torch.load('effnet_rednet_220_60000_npcc.pth', map_location=device))
model.to(device)
model.eval()

# --- Define any transforms  used during training ---
transform = transforms.Compose([
    transforms.ToTensor(),

])

# --- Read video ---
cap = cv2.VideoCapture(VIDEO_PATH)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Loaded video with {frame_count} frames at {fps:.2f} FPS")

# --- Setup video writer for reconstructed output ---
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height), isColor=False)

# --- Setup real-time matplotlib display ---
plt.ion()
fig, ax = plt.subplots()
img_display = ax.imshow(np.zeros((frame_height, frame_width)), cmap='gray', vmin=0, vmax=255)
plt.title("Reconstructed Frames")
plt.tight_layout()

frame_idx = 0
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    input_tensor = transform(gray).unsqueeze(0).to(device)  # shape: [1, 1, H, W]

    with torch.no_grad():
        output = model(input_tensor)

    # Convert output to image
    output_img = output.squeeze().cpu().numpy()
    output_img = (output_img * 255).astype(np.uint8)

    # Convert to 3-channel if needed for compatibility with some players
    video_writer.write(output_img)

    # Real-time matplotlib update
    img_display.set_data(output_img)
    plt.draw()
    plt.pause(0.001)

    frame_idx += 1
    print(f"Current FPS: {frame_idx/(time.time() - start_time)}")
end_time = time.time()
cap.release()
video_writer.release()

# --- FPS Reporting ---
total_time = end_time - start_time
if total_time > 0:
    processing_fps = frame_idx / total_time
    print(f"Processed and saved {frame_idx} reconstructed frames to {OUTPUT_VIDEO_PATH}.")
    print(f"Processing FPS: {processing_fps:.2f}")
else:
    print("Warning: Total processing time is zero. Cannot compute FPS.")
plt.close()
plt.ioff()
plt.show()
