import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import rasterio
from rasterio.windows import Window
import torchvision.transforms as transforms

class TiffImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.mode = mode
        self.files = sorted([os.path.join(root, file) for file in os.listdir(root) if file.endswith('.tiff')])
        
    def chip_image(self, img, size=256):
        """Chip a large image into 256x256 patches"""
        h, w = img.shape[:2]
        patches = []
        
        for i in range(0, h, size):
            for j in range(0, w, size):
                if i + size <= h and j + size <= w:
                    patch = img[i:i+size, j:j+size]
                    patches.append(patch)
        
        return patches

    def __getitem__(self, index):
        # Read the tiff file
        with rasterio.open(self.files[index]) as src:
            img = src.read()  # Read all bands
            img = np.transpose(img, (1, 2, 0))  # Change to HWC format
            
        # Chip the image
        patches = self.chip_image(img)
        
        # For each patch, create a sample
        samples = []
        for patch in patches:
            # Convert to PIL Image
            patch = Image.fromarray(patch)
            
            # Apply transforms
            if self.transform:
                patch = self.transform(patch)
            
            # Create a sample with dummy Y (affine matrix) since we don't have ground truth
            # In real inference, we'll only use the image
            dummy_Y = torch.eye(2, 3)  # Identity matrix as placeholder
            
            sample = {
                "A": patch,
                "B": patch,  # Same image for both A and B in inference
                "Y": dummy_Y
            }
            samples.append(sample)
        
        return samples

    def __len__(self):
        return len(self.files) 