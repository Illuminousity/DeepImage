import re
from PIL import Image
import os
from torch.utils.data import Dataset

class DiffusionDataset(Dataset):
    def __init__(self, diffused_dir, clean_dir, transform=None, cap=60000):
        super().__init__()
        self.diffused_dir = diffused_dir
        self.clean_dir = clean_dir
        self.transform = transform
        
        pattern = re.compile(r'^captured_frame_(\d+)\.png$')
        self.diffused_files = []
        diffused = os.listdir(self.diffused_dir)
        for i in range(int(cap)):
            if pattern.match(diffused[i]):
                self.diffused_files.append(diffused[i])

        
        
        self.diffused_files.sort(key=lambda x: int(pattern.match(x).group(1)))

    def __len__(self):
        return len(self.diffused_files)

    def __getitem__(self, idx):
        diffused_filename = self.diffused_files[idx]
        
        match = re.match(r'^captured_frame_(\d+)\.png$', diffused_filename)
        if not match:
            raise ValueError(
                f"File {diffused_filename} does not match 'captured_frame<number>.png' naming."
            )
        index_str = match.group(1)
        
        raw_filename = f"captured_frame_{index_str}.png"
        
        diffused_path = os.path.join(self.diffused_dir, diffused_filename)
        clean_path    = os.path.join(self.clean_dir, raw_filename)
        
        diffused_img = Image.open(diffused_path).convert('L')
        clean_img    = Image.open(clean_path).convert('L')

        if self.transform:
            diffused_img = self.transform(diffused_img)
            clean_img    = self.transform(clean_img)

        return diffused_img, clean_img