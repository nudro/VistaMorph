import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
from lpips import LPIPS
from torch.cuda.amp import GradScaler, autocast
import antialiased_cnns
from datasets_stn import *
import kornia
from kornia import morphology as morph
import kornia.contrib as K
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--dataset_name", type=str, default="eurecom_warped_pairs", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--experiment", type=str, default="tfcgan_stn_arar", help="experiment name")
parser.add_argument("--order", type=str, default="fBA", help="order of STN inputs")
opt = parser.parse_args()

os.makedirs("./images/test_results/%s_%s" % (opt.experiment, opt.order), exist_ok=True)

# Device configuration for Apple M3
device = torch.device("mps")
torch.manual_seed(42)

# Define tensor types for MPS
FloatTensor = torch.FloatTensor
HalfTensor = torch.HalfTensor

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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        input_shape = (opt.channels, opt.img_height, opt.img_width)
        self.localization = LocalizerVIT(input_shape)
        self.theta_emb = nn.Linear(1, opt.img_height * opt.img_width)

        # Adjusted for patch_size=16 which gives 257 tokens for 256x256 images
        self.fc_loc = nn.Sequential(
            nn.Linear(1*257*768, 1024),
            nn.ReLU(False),
            nn.Linear(1024, 512),
            nn.ReLU(False),
            nn.Linear(512, 256),
            nn.Sigmoid(),
            nn.Linear(256, 3*2))
        self.fc_loc[6].bias.data.copy_(torch.zeros(3*2))

    def stn_phi(self, x):
        xs = self.localization(x)
        print("out Vit:", xs.size())
        xs = xs.view(-1, 1 * xs.size(1) * xs.size(2))
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        return theta

    def forward(self, img_A, img_B, src):
        identity_matrix = [1, 0, 0, 0, 1, 0]

        img_input = torch.cat((img_A, img_B), 1)
        dtheta = self.stn_phi(img_input)
        identity_theta = torch.tensor(identity_matrix, dtype=torch.float).to(device)
        dtheta = dtheta.reshape(img_A.size(0), 2*3)
        dtheta = dtheta + identity_theta.unsqueeze(0).repeat(img_A.size(0),1)

        # get each theta for the batch
        theta_batches = []
        for t in dtheta:
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
            rs_grid = F.affine_grid(theta_batches[i], src_tensors[i].size(), align_corners=True)
            Rs = F.grid_sample(src_tensors[i], rs_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
            warped.append(Rs)

        return torch.cat(warped)

# Initialize models
input_shape = (opt.channels, opt.img_height, opt.img_width)

# Initialize models
model = Net().to(device)
generator1 = GeneratorUNet1(input_shape).to(device)
generator2 = GeneratorUNet2(input_shape).to(device)

if opt.epoch != 0:
    # Load pretrained models
    model.load_state_dict(torch.load("saved_models/%s/net_%d.pth" % (opt.experiment, opt.epoch)))
    generator1.load_state_dict(torch.load("saved_models/%s/generator1_%d.pth" % (opt.experiment, opt.epoch)))
    generator2.load_state_dict(torch.load("saved_models/%s/generator2_%d.pth" % (opt.experiment, opt.epoch)))

# Configure dataloaders
transforms_ = [
    transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

test_dataloader = DataLoader(
    TestImageDataset("../../data/%s" % opt.dataset_name, transforms_=transforms_, mode="test"),
    batch_size=opt.batch_size,
    shuffle=False,
    num_workers=1,
)

# ----------
#  Testing
# ----------

prev_time = time.time()
for i, batch in tqdm(enumerate(test_dataloader)):
    real_A = Variable(batch["A"].type(FloatTensor)).to(device)
    real_B = Variable(batch["B"].type(FloatTensor)).to(device)

    with torch.no_grad():
        fake_B = generator1(real_A)
        warped_fB = model(img_A=real_A, img_B=real_B, src=fake_B)
        T_ = torch.cat(warped_fB)
        if opt.order=='AwB':
            # original from training is below (A, warped_fB)
            warped_B = model(img_A=real_A, img_B=T_, src=real_B)
        elif opt.order=='AfB':
            # swap the order of img inputs to STN for the grid
            warped_B = model(img_A=real_A, img_B=fake_B, src=real_B)
        elif opt.order=='fBA':
            warped_B = model(img_A=fake_B, img_B=real_A, src=real_B)
        elif opt.order=='wBA':
            warped_B = model(img_A=T_, img_B=real_A, src=real_B)

        Reg = torch.cat(warped_B)

    # Save results
    img_sample = torch.cat((real_A.data, real_B.data, fake_B.data, T_.data, Reg.data), -2)
    save_image(img_sample, "./images/test_results/%s_%s/%s_g.png" % (opt.experiment, opt.order, i), nrow=5, normalize=True)

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