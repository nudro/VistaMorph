from PIL import Image
import os
import numpy as np
import cv2
import glob
import torch
import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

    
class ImageDataset(Dataset):
    
    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = transforms.Compose(transforms_)
        # Only get .png files
        self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.png"))
        
        # Get corresponding label files
        self.annots = []
        for img_file in self.files:
            # Get the base filename without extension
            base_name = os.path.splitext(os.path.basename(img_file))[0]
            # Construct path to corresponding label file
            label_path = os.path.join(root, mode, "labels", f"{base_name}.txt")
            self.annots.append(label_path)

        if mode == "test":
            test_files = sorted(glob.glob(os.path.join(root, "test") + "/*.png"))
            self.files.extend(test_files)
            # Add corresponding test label files
            for img_file in test_files:
                base_name = os.path.splitext(os.path.basename(img_file))[0]
                label_path = os.path.join(root, "test", "labels", f"{base_name}.txt")
                self.annots.append(label_path)

    def __getitem__(self, index):
        # Load and process image
        img = Image.open(self.files[index % len(self.files)])
        w, h = img.size
        img_A = img.crop((0, 0, w / 2, h))
        img_B = img.crop((w / 2, 0, w, h))
        
        # Resize images
        newsize = (256, 256)
        img_A = img_A.resize(newsize, Image.Resampling.BICUBIC)
        img_B = img_B.resize(newsize, Image.Resampling.BICUBIC)
    
        # Transform to tensors and normalize
        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        # Load and process label
        with open(self.annots[index % len(self.annots)], 'r') as f:
            label_str = f.read().strip()
            # Split the string and convert to float, taking only first 6 numbers
            label = np.array([float(x) for x in label_str.split()])
            label = label.reshape(3,3)
            label = np.linalg.inv(label)
            label = label.flatten()
            label = label[:6]
    
        return {"A": img_A, 
                "B": img_B,
                "Y": label}

    def __len__(self):
        return len(self.files)
    
    
class TestImageDataset(Dataset):
    
    def __init__(self, root, transforms_=None, mode="test"):
        self.transform = transforms.Compose(transforms_)
        # Only get .png files
        self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.png"))
        
        # Get corresponding label files
        self.annots = []
        for img_file in self.files:
            # Get the base filename without extension
            base_name = os.path.splitext(os.path.basename(img_file))[0]
            # Construct path to corresponding label file
            label_path = os.path.join(root, mode, "labels", f"{base_name}.txt")
            self.annots.append(label_path)

    def __getitem__(self, index):
        # Load and process image
        img = Image.open(self.files[index % len(self.files)])
        w, h = img.size
        img_A = img.crop((0, 0, w / 2, h))
        img_B = img.crop((w / 2, 0, w, h))
        
        # Resize images
        newsize = (256, 256)
        img_A = img_A.resize(newsize, Image.Resampling.BICUBIC)
        img_B = img_B.resize(newsize, Image.Resampling.BICUBIC)
    
        # Transform to tensors and normalize
        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        # Load and process label
        with open(self.annots[index % len(self.annots)], 'r') as f:
            label_str = f.read().strip()
            # Split the string and convert to float, taking only first 6 numbers
            label = np.array([float(x) for x in label_str.split()])
            label = label.reshape(3,3)
            label = np.linalg.inv(label)
            label = label.flatten()
            label = label[:6]
    
        return {"A": img_A, 
                "B": img_B,
                "Y": label}

    def __len__(self):
        return len(self.files)
    
