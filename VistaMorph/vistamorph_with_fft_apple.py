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
import torch.nn.functional as F
from lpips import LPIPS
from torch.cuda.amp import GradScaler, autocast
import antialiased_cnns
from datasets_stn import * # only A and B
import kornia
from lpips import LPIPS
from kornia import morphology as morph
import kornia.contrib as K

"""
Version of VistaMorph with Fourier Transform Loss to handle datasets where there are
dark or low-light visible pairs. No morphological triplet loss. Uses Apple M3 GPU.
"""

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=1, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="eurecom_warped_pairs", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=12, help="size of the batches")
parser.add_argument("--lr", type=float, default=1e-4, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=50, help="interval between sampling of images from generators")
parser.add_argument("--checkpoint_interval", type=int, default=50, help="interval between model checkpoints")
parser.add_argument("--experiment", type=str, default="tfcgan_stn_arar", help="experiment name")
opt = parser.parse_args()

os.makedirs("./images/%s" % opt.experiment, exist_ok=True)
os.makedirs("./saved_models/%s" % opt.experiment, exist_ok=True)
os.makedirs("./LOGS/%s" % opt.experiment, exist_ok=True)

# Device configuration for Apple M3
device = torch.device("mps")
torch.manual_seed(42)

# Define tensor types for MPS
FloatTensor = torch.FloatTensor
HalfTensor = torch.HalfTensor

##########################
# Loss functions
##########################
criterion_GAN = torch.nn.BCEWithLogitsLoss().to(device)

criterion_lpips = LPIPS(
    net_type='vgg',  # choose a network type from ['alex', 'squeeze', 'vgg']
    version='0.1'  # Currently, v0.1 is supported
).to(device)

criterion_L1 = nn.L1Loss().to(device)

# Amplitude and Phase Losses for the FFT Loss
criterion_amp = nn.L1Loss().to(device)
criterion_phase = nn.L1Loss().to(device)

############################
#  Utils
############################

# Calculate output of image discriminator (PatchGAN)
patch = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def expand(tensor):
    # to add 3Channels from 1 channel (handy)
    t = torch.Tensor(tensor).to(device)
    t = t.reshape(t.size(0), 1, t.size(1), t.size(2))
    t = t.expand(-1, 3, -1, -1) # needs to be [1, 3, 256, 256]
    return t

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

        self.fc_loc = nn.Sequential(
            nn.Linear(1*17*768, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.Sigmoid(),
            nn.Linear(256, 3*2))
        self.fc_loc[2].bias.data.zero_() # DO NOT CHANGE!

    def stn_phi(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 1 * xs.size(1) * xs.size(2))
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        return theta

    def forward(self, img_A, img_B, src):
        identity_matrix = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float).to(device)

        with autocast():
            img_input = torch.cat((img_A, img_B), 1)
            dtheta = self.stn_phi(img_input)
            dtheta = dtheta.reshape(img_A.size(0), 2*3)
            dtheta = dtheta + identity_matrix.unsqueeze(0).repeat(img_A.size(0),1)

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
                Rs = F.grid_sample(src_tensors[i], rs_grid, mode='bicubic', padding_mode='border', align_corners=True)
                warped.append(Rs.type(HalfTensor))

        return torch.cat(warped)

# ... [Previous UNet and Generator code remains the same] ...

class FFT_Components(object):
    def __init__(self, image):
        self.image = image

    def make_components(self):
        img = np.array(self.image)
        f_result = np.fft.rfft2(img)
        fshift = np.fft.fftshift(f_result)
        amp = np.abs(fshift)
        phase = np.arctan2(fshift.imag,fshift.real)
        return amp, phase

    def make_spectra(self):
        img = np.array(self.image)
        f_result = np.fft.fft2(img)
        fshift = np.fft.fftshift(f_result)
        magnitude_spectrum = np.log(np.abs(fshift))
        return magnitude_spectrum

def fft_components(thermal_tensor):
    AMP = []
    PHA = []
    for t in range(0, opt.batch_size):
        b = transforms.ToPILImage()(thermal_tensor[t, :, :, :]).convert("L")
        fft_space = FFT_Components(b)
        amp, phase = torch.Tensor(fft_space.make_components()).to(device)
        AMP.append(amp)
        PHA.append(phase)

    AMP_tensor = torch.cat(AMP).reshape(opt.batch_size, 1, opt.img_height, 129)
    PHA_tensor = torch.cat(PHA).reshape(opt.batch_size, 1, opt.img_height, 129)
    return AMP_tensor, PHA_tensor

# Initialize models
input_shape = (opt.channels, opt.img_height, opt.img_width)

# Initialize models
model = Net().to(device)
generator1 = GeneratorUNet1(input_shape).to(device)
generator2 = GeneratorUNet2(input_shape).to(device)
discriminator1 = Discriminator1(input_shape).to(device)
discriminator2 = Discriminator2(input_shape).to(device)

if opt.epoch != 0:
    # Load pretrained models
    model.load_state_dict(torch.load("saved_models/%s/net_%d.pth" % (opt.experiment, opt.epoch)))
    generator1.load_state_dict(torch.load("saved_models/%s/generator1_%d.pth" % (opt.experiment, opt.epoch)))
    generator2.load_state_dict(torch.load("saved_models/%s/generator2_%d.pth" % (opt.experiment, opt.epoch)))
    discriminator1.load_state_dict(torch.load("saved_models/%s/discriminator1_%d.pth" % (opt.experiment, opt.epoch)))
    discriminator2.load_state_dict(torch.load("saved_models/%s/discriminator2_%d.pth" % (opt.experiment, opt.epoch)))
else:
    # Initialize weights
    model.apply(weights_init_normal)
    generator1.apply(weights_init_normal)
    generator2.apply(weights_init_normal)
    discriminator1.apply(weights_init_normal)
    discriminator2.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(generator1.parameters(), generator2.parameters(), model.parameters()),
    lr=opt.lr,
    betas=(opt.b1, opt.b2),
)
optimizer_D = torch.optim.Adam(
    itertools.chain(discriminator1.parameters(), discriminator2.parameters()),
    lr=opt.lr,
    betas=(opt.b1, opt.b2),
)

# Configure dataloaders
transforms_ = [
    transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

dataloader = DataLoader(
    ImageDataset("../../data/%s" % opt.dataset_name, transforms_=transforms_),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

# ----------
#  Training
# ----------

prev_time = time.time()
scaler = GradScaler()

for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        # Model inputs
        real_A = Variable(batch["A"].type(FloatTensor)).to(device)
        real_B = Variable(batch["B"].type(FloatTensor)).to(device)

        # Adversarial ground truths with label smoothing
        valid = Variable(FloatTensor(np.ones((real_A.size(0), *patch))), requires_grad=False).to(device)
        fake = Variable(FloatTensor(np.zeros((real_A.size(0), *patch))), requires_grad=False).to(device)

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        with autocast():
            # Generate fake samples
            fake_B = generator1(real_A)
            warped_B = model(img_A=real_A, img_B=fake_B, src=real_B)
            fake_A = generator2(warped_B)

            # Reconstruction loss
            recon_loss = criterion_L1(warped_B, fake_B)

            # Perceptual loss
            perc_A = criterion_lpips(fake_A, real_A)
            perc_B = criterion_lpips(fake_B, real_B)
            perc_loss = (perc_A + perc_B).mean()

            # Fourier Transform Loss
            Amp_f, Pha_f = fft_components(fake_A)
            Amp_r, Pha_r = fft_components(real_A)
            loss_Amp = criterion_amp(Amp_f, Amp_r)
            loss_Pha = criterion_phase(Pha_f, Pha_r)
            loss_FFT = (loss_Amp + loss_Pha).mean()

            # Total loss
            loss_G = (recon_loss + perc_loss + loss_FFT).mean()

        scaler.scale(loss_G).backward()
        scaler.step(optimizer_G)

        # -----------------------
        #  Train Discriminator
        # -----------------------

        optimizer_D.zero_grad()

        with autocast():
            # Real loss
            pred_real = discriminator1(real_A, real_B)
            loss_real = criterion_GAN(pred_real, valid)

            # Fake loss
            pred_fake = discriminator1(real_A, fake_B.detach())
            loss_fake = criterion_GAN(pred_fake, fake)

            # Total discriminator loss
            loss_D = 0.5 * (loss_real + loss_fake)

        scaler.scale(loss_D).backward()
        scaler.step(optimizer_D)

        scaler.update()

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] ETA: %s"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(dataloader),
                loss_D.item(),
                loss_G.item(),
                time_left,
            )
        )

        # If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done)

        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(model.state_dict(), "saved_models/%s/net_%d.pth" % (opt.experiment, epoch))
            torch.save(generator1.state_dict(), "saved_models/%s/generator1_%d.pth" % (opt.experiment, epoch))
            torch.save(generator2.state_dict(), "saved_models/%s/generator2_%d.pth" % (opt.experiment, epoch))
            torch.save(discriminator1.state_dict(), "saved_models/%s/discriminator1_%d.pth" % (opt.experiment, epoch))
            torch.save(discriminator2.state_dict(), "saved_models/%s/discriminator2_%d.pth" % (opt.experiment, epoch)) 