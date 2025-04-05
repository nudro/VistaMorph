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
from lpips import LPIPS
import cv2
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import antialiased_cnns
from datasets_stn import *
import timm

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--dataset_name", type=str, default="facades", help="name of the dataset")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--experiment", type=str, default="none", help="experiment name")
parser.add_argument("--order", type=str, default="AwB", help="AwB, AfB, fBA, wBA")
opt = parser.parse_args()

os.makedirs("./images/test_results/%s_%s" % (opt.experiment, opt.order), exist_ok=True)

# Device configuration
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Configure Tensor type
Tensor = torch.FloatTensor

###################
# UNET for STN
###################

class UNetDownLoc(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDownLoc, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUpLoc(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUpLoc, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
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


class LocalizerUNet(nn.Module):
    def __init__(self, img_shape):
        super(LocalizerUNet, self).__init__()
        channels, self.h, self.w = img_shape
        self.down1 = UNetDownLoc(channels*2, 64)
        self.down2 = UNetDownLoc(64, 128)
        self.down3 = UNetDownLoc(128, 256)
        self.down4 = UNetDownLoc(256, 512)
        self.down5 = UNetDownLoc(512, 512)
        self.down6 = UNetDownLoc(512, 512)

        self.up1 = UNetUpLoc(512, 512)
        self.up2 = UNetUpLoc(1024, 512)
        self.up3 = UNetUpLoc(1024, 256)
        self.up4 = UNetUpLoc(512, 128)
        self.up5 = UNetUpLoc(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, channels, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
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
        out = self.final(u5)
        return out


#################
# STN
#################

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        input_shape = (opt.channels, opt.img_height, opt.img_width)
        self.localization = LocalizerUNet(input_shape)

        self.fc_loc = nn.Sequential(
            nn.Linear(3*256*256, 256),
            nn.ReLU(True),
            nn.Linear(256, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].bias.data.zero_()

    def stn_phi(self, x):
        xs = self.localization(x)  # convolves the cat channel=6
        xs = xs.view(-1, 3*256*256)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        return theta

    def forward(self, img_A, img_B, src):
        with autocast(device_type='mps' if torch.backends.mps.is_available() else 'cpu', dtype=torch.float32):
            img_input = torch.cat((img_A, img_B), 1)
            dtheta = self.stn_phi(img_input)  # deformation field for real A and real B
            identity_theta = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float).to(device)
            dtheta = dtheta.reshape(img_B.size(0), 2*3)
            theta = dtheta + identity_theta.unsqueeze(0).repeat(img_B.size(0), 1)

            theta_batches = []
            for t in theta:
                this_theta = (t.view(-1, 2, 3)).reshape(1,2,3)
                theta_batches.append(this_theta)

            src_tensors = []
            for img in src:
                this_img_src = img.reshape(1, img.size(0), img.size(1), img.size(2))
                src_tensors.append(this_img_src)

            warped = []
            for i in range(len(src_tensors)):
                rs_grid = F.affine_grid(theta_batches[i], src_tensors[i].size(), align_corners=True)
                Rs = F.grid_sample(src_tensors[i], rs_grid, mode='nearest', padding_mode='border', align_corners=True)
                warped.append(Rs.type(Tensor).to(device))

        return warped


##############################
#     Generator  U-NET
##############################

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 1, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2, inplace=False))
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
            nn.ReLU(inplace=False),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x


class UNetDownNOBP(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDownLoc, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUpNOBP(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUpLoc, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
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


class GeneratorUNet(nn.Module):
    def __init__(self, img_shape):
        super(GeneratorUNet, self).__init__()
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
        output = self.final(u5)
        return output


##########
# UTILS
###########

def load_clean_state(model_name, checkpoint_path):
    from collections import OrderedDict
    state_dict = torch.load(checkpoint_path, map_location=device)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v

    # load params
    model_name.load_state_dict(new_state_dict)
    print(f"Loaded successfully {model_name} state dict")


##############################
#       Initialize
##############################
input_shape_global = (opt.channels, opt.img_height, opt.img_width)
generator = GeneratorUNet(input_shape_global)
generator = generator.to(device)
g_path = "./saved_models/%s/generator_%d.pth" % (opt.experiment, opt.epoch)
load_clean_state(generator, g_path)

model = Net().to(device)
m_path = "./saved_models/%s/stn_%d.pth" % (opt.experiment, opt.epoch)
load_clean_state(model, m_path)

##############################
# Transforms and Dataloaders
##############################

# Resizing happens in the ImageDataset() to 256 x 256 so that I can get patches
transforms_ = [
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

test_dataloader = DataLoader(
    TestImageDataset(root="./data/%s" % opt.dataset_name,
        transforms_=transforms_,
        mode="test"),
    batch_size=1,
    shuffle=False,
    num_workers=1,
)


##############################
#       Testing
##############################
for i, batch in tqdm(enumerate(test_dataloader)):
    real_A = Variable(batch["A"]).type(Tensor).to(device)
    real_B = Variable(batch["B"]).type(Tensor).to(device)

    with torch.no_grad():
        with autocast(device_type='mps' if torch.backends.mps.is_available() else 'cpu', dtype=torch.float32):
            fake_B = generator(real_A)
            warped_fB = model(img_A=real_A, img_B=real_B, src=fake_B)
            T_ = torch.cat(warped_fB)
            if opt.order == 'AwB':
                # original from training is below (A, warped_fB)
                warped_B = model(img_A=real_A, img_B=T_, src=real_B)
            elif opt.order == 'AfB':
                # swap the order of img inputs to STN for the grid
                warped_B = model(img_A=real_A, img_B=fake_B, src=real_B)
            elif opt.order == 'fBA':
                warped_B = model(img_A=fake_B, img_B=real_A, src=real_B)
            elif opt.order == 'wBA':
                warped_B = model(img_A=T_, img_B=real_A, src=real_B)

            Reg = torch.cat(warped_B)

    # GLOBAL
    img_sample_global = torch.cat((real_A.data, real_B.data, fake_B.data, T_.data, Reg.data), -2)
    save_image(img_sample_global, "./images/test_results/%s_%s/%s_g.png" % (opt.experiment, opt.order, i), nrow=5, normalize=True)


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=(224, 224), patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x 