import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
import cv2
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import antialiased_cnns
import rasterio
from rasterio.windows import Window
import kornia
import kornia.contrib as K
from PIL import Image
import logging
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=200, help="epoch to load")
parser.add_argument("--input_dir", type=str, default="./input", help="Directory containing input tiff files")
parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save output theta files")
parser.add_argument("--weights_dir", type=str, default="./weights", help="Directory containing model weights")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--experiment", type=str, default="spacenet_chippedV1FFT", help="experiment name")
opt = parser.parse_args()

# Create output directory with error handling
try:
    os.makedirs(opt.output_dir, exist_ok=True)
    logger.info(f"Output directory created/verified at: {opt.output_dir}")
except Exception as e:
    logger.error(f"Failed to create output directory: {str(e)}")
    raise

# Verify input directory exists
if not os.path.exists(opt.input_dir):
    logger.error(f"Input directory does not exist: {opt.input_dir}")
    raise FileNotFoundError(f"Input directory not found: {opt.input_dir}")

# Verify weights directory exists
if not os.path.exists(opt.weights_dir):
    logger.error(f"Weights directory does not exist: {opt.weights_dir}")
    raise FileNotFoundError(f"Weights directory not found: {opt.weights_dir}")

cuda = True if torch.cuda.is_available() else False

# Set fixed random number seed
torch.manual_seed(42)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

#################
# ViT for STN
################


class LocalizerVIT(nn.Module):
    def __init__(self, img_shape):
        super(LocalizerVIT, self).__init__()
        channels, self.h, self.w = img_shape
        self.vit = nn.Sequential(
            K.VisionTransformer(image_size=self.h, patch_size=64, in_channels=channels*2) # (A,B), changed from 16-> 32 patch
        )

    def forward(self, x):
        # if you try different patch sizes, adjust tensors
        # patch 16: torch.Size([batch, 257, 768])
        # patch 32: torch.Size([batch, 65, 768])
        # patch 64: torch.Size([batch, 17, 768])
        with autocast():
            out = self.vit(x).type(HalfTensor) # returns at patch_size = 16, torch.Size([batch, 257, 768])
            print("out Vit:", out.size())
        return out

####################################################################
# STN (Jadeerburg, 2015)
# Adopted from https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html
####################################################################


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        input_shape = (opt.channels, opt.img_height, opt.img_width)
        self.localization = LocalizerVIT(input_shape)
        self.theta_emb = nn.Linear(1, opt.img_height * opt.img_width)

        # nn.Linear(1*257*768, 1024), # (hard-coded in, 257, 768 based on ViT output)
        self.fc_loc = nn.Sequential(
            nn.Linear(1*17*768, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512), # Added more layers, will this stop the affine warp?
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.Sigmoid(), # <--- SIGMOID will this force the matrix values between [-1,+1]
            nn.Linear(256, 3*2))
        self.fc_loc[2].bias.data.zero_() # DO NOT CHANGE!

    def stn_phi(self, x):
        xs = self.localization(x) # (A,B), 6 ch
        xs = xs.view(-1, 1 * xs.size(1) * xs.size(2))
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        return theta

    def forward(self, img_A, img_B, src):
        identity_matrix = [1, 0, 0, 0, 1, 0]

        with autocast():
            img_input = torch.cat((img_A, img_B), 1)
            dtheta = self.stn_phi(img_input)
            identity_theta = torch.tensor(identity_matrix, dtype=torch.float).cuda()
            dtheta = dtheta.reshape(img_A.size(0), 2*3)
            theta = dtheta + identity_theta.unsqueeze(0).repeat(img_A.size(0),1)

            # get each theta for the batch
            theta_batches = []
            for t in theta:
                this_theta = (t.view(-1, 2, 3)).reshape(1,2,3)
                theta_batches.append(this_theta)

            # just the tensors of the source image to be aligned
            src_tensors = []
            for img in src:
                this_img_src = img.reshape(1, img.size(0), img.size(1), img.size(2))
                src_tensors.append(this_img_src)

            # result
            warped = []
            for i in range(len(src_tensors)):
                # Adjusting some of the parameters like align_cornrs, mode, and padding_mode will change output
                # Based on my experiments, the below are the best configurations
                rs_grid = F.affine_grid(theta_batches[i], src_tensors[i].size(), align_corners=True)
                Rs = F.grid_sample(src_tensors[i], rs_grid,  mode='bicubic', padding_mode='border', align_corners=True)
                warped.append(Rs.type(HalfTensor))

        return torch.cat(warped), theta  # Now returning both the warped image and the affine matrix

##############################
# 2 Generators
# Based on  my TFC-GAN implementation from ICIP 2023 - https://arxiv.org/pdf/2302.09395.pdf
##############################


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 1, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(antialiased_cnns.BlurPool(out_size, stride=2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
                nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
                antialiased_cnns.BlurPool(out_size, stride=1),
                nn.InstanceNorm2d(out_size),
                nn.ReLU(inplace=True),
            ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x


class GeneratorUNet1(nn.Module):
    def __init__(self, img_shape):
        super(GeneratorUNet1, self).__init__()
        channels, self.h, self.w = img_shape
        self.down1 = UNetDown(channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256, dropout=0.5)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, normalize=False)
        self.down6 = UNetDown(512, 512)

        self.up1 = UNetUp(512, 512)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 256, dropout=0.5)
        self.up4 = UNetUp(512, 128)
        self.up5 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        with autocast():
            d1 = self.down1(x)
            d2 = self.down2(d1)
            d3 = self.down3(d2)
            d4 = self.down4(d3)
            d5 = self.down5(d4)
            d6 = self.down6(d5)
            u1 = self.up1(d6, d5)
            u2 = self.up2(u1, d4)
            u3 = self.up3(u2, d3)
            u4 = self.up4(u3, d2)
            u5 = self.up5(u4, d1)
            output = self.final(u5).type(HalfTensor)
        return output


class GeneratorUNet2(nn.Module):
    def __init__(self, img_shape):
        super(GeneratorUNet2, self).__init__()
        channels, self.h, self.w = img_shape
        self.down1 = UNetDown(channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256, dropout=0.5)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, normalize=False)
        self.down6 = UNetDown(512, 512)

        self.up1 = UNetUp(512, 512)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 256, dropout=0.5)
        self.up4 = UNetUp(512, 128)
        self.up5 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        with autocast():
            d1 = self.down1(x)
            d2 = self.down2(d1)
            d3 = self.down3(d2)
            d4 = self.down4(d3)
            d5 = self.down5(d4)
            d6 = self.down6(d5)
            u1 = self.up1(d6, d5)
            u2 = self.up2(u1, d4)
            u3 = self.up3(u2, d3)
            u4 = self.up4(u3, d2)
            u5 = self.up5(u4, d1)
            output = self.final(u5).type(HalfTensor)
        return output

#=================================
# Configure data loader for tiff files
class TiffImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode="test"):
        self.transform = transforms.Compose(transforms_)
        self.mode = mode
        self.files = sorted([os.path.join(root, file) for file in os.listdir(root) if file.endswith('.tiff')])
        self.patches = []
        self.patch_to_file = []  # Maps patch index to file index
        
        # Pre-process all files to get patches
        for file_idx, file_path in enumerate(self.files):
            try:
                with rasterio.open(file_path) as src:
                    img = src.read()  # Read all bands
                    img = np.transpose(img, (1, 2, 0))  # Change to HWC format
                    
                # Get patches for this file
                file_patches = self.chip_image(img)
                self.patches.extend(file_patches)
                self.patch_to_file.extend([file_idx] * len(file_patches))
            except Exception as e:
                print(f"Error processing file {file_path}: {str(e)}")
                continue
        
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
        # Get the patch
        patch = self.patches[index]
        
        # Convert to PIL Image
        patch = Image.fromarray(patch)
        
        # Apply transforms
        if self.transform:
            patch = self.transform(patch)
        
        # Create a sample with dummy Y (affine matrix) since we don't have ground truth
        dummy_Y = torch.eye(2, 3)  # Identity matrix as placeholder
        
        return {
            "A": patch,
            "B": patch,  # Same image for both A and B in inference
            "Y": dummy_Y,
            "file_idx": self.patch_to_file[index]  # Keep track of which file this patch came from
        }

    def __len__(self):
        return len(self.patches)

transforms_ = [
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

test_dataloader = DataLoader(
    TiffImageDataset(
        root=opt.input_dir,
        transforms_=transforms_,
        mode="test"
    ),
    batch_size=1,
    shuffle=False,
    num_workers=1,
)

# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
HalfTensor = torch.cuda.HalfTensor if cuda else torch.HalfTensor

# Loss functions
criterion_pixelwise = torch.nn.MSELoss()

# Initialize generator and discriminator
input_shape = (opt.channels, opt.img_height, opt.img_width)

# Initialize networks
net = Net()
generator1 = GeneratorUNet1(input_shape)
generator2 = GeneratorUNet2(input_shape)

if cuda:
    net = net.cuda()
    generator1 = generator1.cuda()
    generator2 = generator2.cuda()
    criterion_pixelwise.cuda()

# ======
def load_clean_state(model_name, checkpoint_path):
    try:
        from collections import OrderedDict
        state_dict = torch.load(checkpoint_path)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
            
        # load params
        model_name.load_state_dict(new_state_dict)
        logger.info(f"Successfully loaded state dict for {model_name.__class__.__name__}")
    except Exception as e:
        logger.error(f"Failed to load weights from {checkpoint_path}: {str(e)}")
        raise

# Load pretrained models
net_path = os.path.join(opt.weights_dir, f"net_{opt.epoch}.pth")
g1_path = os.path.join(opt.weights_dir, f"generator1_{opt.epoch}.pth")
g2_path = os.path.join(opt.weights_dir, f"generator2_{opt.epoch}.pth")

# Check if weights exist
for path in [net_path, g1_path, g2_path]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required weight file not found: {path}")

load_clean_state(net, net_path)
load_clean_state(generator1, g1_path)
load_clean_state(generator2, g2_path)

# Set to eval mode
net.eval()
generator1.eval()
generator2.eval()

logger.info("Starting inference...")

# ----------
#  Testing
# ----------
# Create a list to store all predictions
all_predictions = []

for i, sample in enumerate(test_dataloader):
    try:
        real_A = Variable(sample["A"].type(Tensor))
        real_B = Variable(sample["B"].type(Tensor))
        file_idx = sample["file_idx"].item()
        
        with torch.no_grad(): 
            fake_B = generator1(real_A)
            warped_B, theta = net(img_A=real_A, img_B=fake_B, src=real_B) 
            fake_A = generator2(warped_B)
            
            # Get the predicted affine matrix
            theta_np = theta.cpu().numpy()
            
            # Format: a b tx c d ty 0 0 1
            formatted_theta = f"{theta_np[0][0]:.6f} {theta_np[0][1]:.6f} {theta_np[0][2]:.6f} {theta_np[0][3]:.6f} {theta_np[0][4]:.6f} {theta_np[0][5]:.6f} 0 0 1"
            
            # Add to predictions list
            all_predictions.append({
                'image_id': f"image_{file_idx:05d}_patch_{i:05d}",
                'affine_matrix': formatted_theta
            })
            
            logger.info(f"Processed patch {i} from file {file_idx}")
    except Exception as e:
        logger.error(f"Error processing patch {i} from file {file_idx}: {str(e)}")
        continue

# Create solution.csv
try:
    solution_df = pd.DataFrame(all_predictions)
    solution_path = os.path.join(opt.output_dir, 'solution.csv')
    solution_df.to_csv(solution_path, index=False)
    logger.info(f"Solution saved to {solution_path}")
    
    # Verify the file was created and is readable
    if not os.path.exists(solution_path):
        raise FileNotFoundError(f"Solution file was not created at {solution_path}")
    
    # Verify the file is not empty
    if os.path.getsize(solution_path) == 0:
        raise ValueError("Solution file was created but is empty")
        
    logger.info(f"Solution file verified: {solution_path}")
except Exception as e:
    logger.error(f"Failed to create solution.csv: {str(e)}")
    raise

logger.info("Inference completed successfully")

