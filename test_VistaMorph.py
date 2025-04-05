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
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
import cv2
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import antialiased_cnns
from datasets_stn import * 
import kornia
import torch.nn.contrib as K

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to load")
parser.add_argument("--dataset_name", type=str, default="eurecom_warped_pairs", help="name of the dataset")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--experiment", type=str, default="tfcgan_stn", help="experiment name")
opt = parser.parse_args()

os.makedirs("./images/test_results/%s" % opt.experiment, exist_ok=True)
cuda = True if torch.cuda.is_available() else False

# Set fixed random number seed
torch.manual_seed(42)

# Configure data loader
transforms_ = [
    transforms.Resize((opt.img_height, opt.img_width), transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

test_dataloader = DataLoader(
    TestImageDataset(
        root="./data/%s" % opt.dataset_name,
        transforms_=transforms_,
        mode="test"
    ),
    batch_size=1,
    shuffle=True,
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

# Load pretrained models
net.load_state_dict(torch.load("saved_models/%s/net_%d.pth" % (opt.experiment, opt.epoch)))
generator1.load_state_dict(torch.load("saved_models/%s/generator1_%d.pth" % (opt.experiment, opt.epoch)))
generator2.load_state_dict(torch.load("saved_models/%s/generator2_%d.pth" % (opt.experiment, opt.epoch)))

# Set to eval mode
net.eval()
generator1.eval()
generator2.eval()

def sample_images(batches_done):
    """Saves a generated sample from the validation set"""
    imgs = next(iter(test_dataloader))
    real_A = Variable(imgs["A"].type(Tensor))
    real_B = Variable(imgs["B"].type(Tensor))
    
    with torch.no_grad():
        warped = net(real_A, real_B, [real_A])
        fake_B = generator1(warped)
        fake_A = generator2(fake_B)
        
        # Save image grid
        img_sample = torch.cat((real_A, warped, fake_B, fake_A, real_B), -2)
        save_image(img_sample, "images/test_results/%s/%s.png" % (opt.experiment, batches_done), nrow=5, normalize=True)

# ----------
#  Testing
# ----------

prev_time = time.time()
for i, batch in enumerate(test_dataloader):
    # Model inputs
    real_A = Variable(batch["A"].type(Tensor))
    real_B = Variable(batch["B"].type(Tensor))
    
    # Generate output
    with torch.no_grad():
        warped = net(real_A, real_B, [real_A])
        fake_B = generator1(warped)
        fake_A = generator2(fake_B)
        
        # Save sample images
        if i % 50 == 0:
            sample_images(i)
            
    # Print time statistics
    batches_done = i
    time_left = datetime.timedelta(seconds=time.time() - prev_time)
    prev_time = time.time()
    
    sys.stdout.write(
        "\r[Batch %d/%d] [Time: %s]"
        % (
            i,
            len(test_dataloader),
            time_left,
        )
    ) 