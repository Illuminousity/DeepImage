import os
import shutil
import re

# Root directory containing the GRIT folders
root_dir = "./DMD/"  # <-- Change this to your actual path
grit_folders = ["220 GRIT", "600 GRIT", "1500 GRIT", "Raw"]
destination_folder = os.path.join(root_dir, "Excess Images")

# Create destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Regex to extract frame number
pattern = re.compile(r"captured_frame_(\d+)\.png")

for folder in grit_folders:
    full_path = os.path.join(root_dir, folder)
    dest_folder = os.path.join(destination_folder,folder)
    for filename in os.listdir(full_path):
        match = pattern.match(filename)
        if match:
            frame_num = int(match.group(1))
            if frame_num > 60000:
                src = os.path.join(full_path, filename)
                dest = os.path.join(dest_folder, f"{folder.replace(' ', '_')}_{filename}")
                shutil.move(src, dest)
                print(f"Moved: {src} → {dest}")
