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
from torch.utils.data import DataLoader, DistributedSampler
from torch.autograd import Variable
import torch.nn.functional as F
from lpips import LPIPS
from torch.distributed import Backend
from torch.cuda.amp import GradScaler, autocast
import antialiased_cnns
from datasets_stn import * # only A and B
import kornia
from lpips import LPIPS
from kornia import morphology as morph
import kornia.contrib as K


"""
Official Implementation of "VISTA-MORPH: UNSUPERVISED IMAGE REGISTRATION
OF VISIBLE-THERMAL FACIAL PAIRS" (https://arxiv.org/pdf/2306.06505.pdf)

Quick Algorithm Overview:

Losses are: Loss_GAN + alpha2*recon_loss + perc_loss + morph_loss

fake_B = G1(A)
fake_A = G2(B)
warped_B = model(A, fake_A, real_B)
L1(A,fake_A)
LPIPS(A, fake_A) + LPIPS(B, fake_B)
morph_triplet(warped_B, real_A, real_B)
Use a ViT 64 patch size for STN localization network

Sigmoid on ViT MLP to push values -1,+1
A->G1->fake_B,A --> phi(real_B) ~ sample -> real_B^ [registered] -> G2 -> fake_A <---> [L1] with real_A

If you are working with dark/no-light thermal pairs, then try the vistamorph_with_fft.py
This includes a Fourier Transform Loss as mentioned in the paper

"""

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=210, help="number of epochs of training")
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
parser.add_argument("--gpu_num", type=int, default=0, help="gpu card")
parser.add_argument("--experiment", type=str, default="tfcgan_stn_arar", help="experiment name")
opt = parser.parse_args()

os.makedirs("./images/%s" % opt.experiment, exist_ok=True)
os.makedirs("./saved_models/%s" % opt.experiment, exist_ok=True)
os.makedirs("./LOGS/%s" % opt.experiment, exist_ok=True)

cuda = True if torch.cuda.is_available() else False
torch.cuda.set_device(opt.gpu_num)
torch.manual_seed(42)


##########################
# Loss functions
##########################
criterion_GAN = torch.nn.BCEWithLogitsLoss() # Relativistic
criterion_lpips = LPIPS(
    net_type='vgg',  # choose a network type from ['alex', 'squeeze', 'vgg']
    version='0.1'  # Currently, v0.1 is supported
)
criterion_L1 = nn.L1Loss()
criterion_morph = nn.TripletMarginLoss(margin=1.0, p=2)

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
    t = torch.Tensor(tensor).cuda()
    t = t.reshape(t.size(0), 1, t.size(1), t.size(2))
    t = t.expand(-1, 3, -1, -1) # needs to be [1, 3, 256, 256]
    return t


def sample_images(batches_done):
    imgs = next(iter(test_dataloader)) # batch_size = 1
    real_A = Variable(imgs["A"].type(HalfTensor)) # torch.Size([1, 3, 256, 256])
    real_B = Variable(imgs["B"].type(HalfTensor))
    fake_B = generator1(real_A)
    fake_A1 = generator2(real_B)
    # pass to generator 2 for fake_A
    warped_B = model(img_A=real_A, img_B=fake_A1, src=real_B)
    img_sample_global = torch.cat((real_A.data, real_B.data, warped_B.data, fake_A1.data, fake_B.data), -1)
    save_image(img_sample_global, "./images/%s/%s.png" % (opt.experiment, batches_done), nrow=4, normalize=True)

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
                # Adjusting some of the parameters like align_cornrs, mode, and padding_mode will change output
                # Based on my experiments, the below are the best configurations
                rs_grid = F.affine_grid(theta_batches[i], src_tensors[i].size(), align_corners=True)
                Rs = F.grid_sample(src_tensors[i], rs_grid,  mode='bicubic', padding_mode='border', align_corners=True)
                warped.append(Rs.type(HalfTensor))

        return torch.cat(warped)

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


##############################
# 2 Discriminators
##############################

class Discriminator1(nn.Module):

    def __init__(self, img_shape):
        super(Discriminator1, self).__init__()
        channels, self.h, self.w = img_shape

        def discriminator_block(in_filters, out_filters):
            layers = [torch.nn.utils.parametrizations.spectral_norm(
                nn.Conv2d(in_filters, out_filters, 4, stride=1, padding=1))] # changed to stride=1 instead of 2
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(antialiased_cnns.BlurPool(out_filters, stride=2)) #blurpool downsample stride=2
            return layers

        self.model = nn.Sequential(
            *discriminator_block((channels * 2), 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        with autocast():
            img_input = torch.cat((img_A, img_B), 1)
            d_in = img_input
            output = self.model(d_in)
        return output.type(HalfTensor)


class Discriminator2(nn.Module):

    def __init__(self, img_shape):
        super(Discriminator2, self).__init__()
        channels, self.h, self.w = img_shape

        def discriminator_block(in_filters, out_filters):
            layers = [torch.nn.utils.parametrizations.spectral_norm(
                nn.Conv2d(in_filters, out_filters, 4, stride=1, padding=1))] # changed to stride=1 instead of 2
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(antialiased_cnns.BlurPool(out_filters, stride=2)) #blurpool downsample stride=2

            return layers

        self.model = nn.Sequential(
            *discriminator_block((channels * 2), 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        with autocast():
            img_input = torch.cat((img_A, img_B), 1)
            d_in = img_input
            output = self.model(d_in)
        return output.type(HalfTensor)

#####################
# LOSSES
#####################


def morph_triplet(real_A, real_B, reg_B):
    kernel = torch.tensor([[0, 1, 0],[1, 1, 1],[0, 1, 0]]).cuda()
    # apply mogrphological grad using kernel
    m_A = 1. - (morph.gradient(real_A, kernel)) # Morphological gradient
    m_B = 1. - (morph.gradient(real_B, kernel)) # Morphological gradient
    m_GB = 1. - (morph.gradient(reg_B, kernel)) # Morphological gradient

    loss = criterion_morph(m_GB, m_A, m_B) # anc, pos, neg
    return loss


def global_pixel_loss(real_B, fake_B):
    loss_pix = criterion_L1(fake_B, real_B)
    return loss_pix


def global_gen_loss(real_A, real_B, fake_img, mode):
    if mode=='B':
        pred_fake = discriminator1(fake_img, real_A)
        real_pred = discriminator1(real_B, real_A)
        loss_GAN = criterion_GAN(pred_fake - real_pred.detach(), valid)
    elif mode=='A':
        pred_fake = discriminator2(fake_img, real_B)
        real_pred = discriminator2(real_A, real_B)
        loss_GAN = criterion_GAN(pred_fake - real_pred.detach(), valid)
    return loss_GAN


def global_disc_loss(real_A, real_B, fake_img, mode):
    if mode=='B':
        pred_real = discriminator1(real_B, real_A)
        pred_fake = discriminator1(fake_img.detach(), real_A)
        loss_real = criterion_GAN(pred_real - pred_fake, valid)
        loss_fake = criterion_GAN(pred_fake - pred_real, fake)
        loss_D = 0.25*(loss_real + loss_fake)

    if mode=='A':
        pred_real = discriminator2(real_A, real_B)
        pred_fake = discriminator2(fake_img.detach(), real_B)
        loss_real = criterion_GAN(pred_real - pred_fake, valid)
        loss_fake = criterion_GAN(pred_fake - pred_real, fake)
        loss_D = 0.25*(loss_real + loss_fake)

    return loss_D


# ===========================================================
# Initialize generator and discriminator
# ===========================================================
input_shape_patch = (opt.channels, opt.img_height, opt.img_width)
generator1 = GeneratorUNet1(input_shape_patch) # for fake_B
generator2 = GeneratorUNet2(input_shape_patch) # for fake_A
discriminator1 = Discriminator1(input_shape_patch) # for fake_B
discriminator2 = Discriminator2(input_shape_patch) # for fake_A

if cuda:
    generator1 = generator1.cuda()
    generator2 = generator2.cuda()
    discriminator1 = discriminator1.cuda()
    discriminator2 = discriminator2.cuda()
    model = Net().cuda()

    criterion_GAN.cuda()
    criterion_lpips.cuda()
    criterion_L1.cuda()
    criterion_morph.cuda()

# Trained on multigpus - change your device ids if needed
generator1 = torch.nn.DataParallel(generator1, device_ids=[0,1])
generator2 = torch.nn.DataParallel(generator2, device_ids=[0,1])
discriminator1 = torch.nn.DataParallel(discriminator1, device_ids=[0,1])
discriminator2 = torch.nn.DataParallel(discriminator2, device_ids=[0,1])
model = torch.nn.DataParallel(model, device_ids=[0,1])


if opt.epoch != 0:
    # Load pretrained models
    generator1.load_state_dict(torch.load("./saved_models/%s/generator1_%d.pth" % (opt.experiment, opt.epoch)))
    generator2.load_state_dict(torch.load("./saved_models/%s/generator2_%d.pth" % (opt.experiment, opt.epoch)))
    discriminator1.load_state_dict(torch.load("./saved_models/%s/discriminator1_%d.pth" % (opt.experiment, opt.epoch)))
    discriminator2.load_state_dict(torch.load("./saved_models/%s/discriminator2_%d.pth" % (opt.experiment, opt.epoch)))
    model.load_state_dict(torch.load("./saved_models/%s/stn_%d.pth" % (opt.experiment, opt.epoch)))

else:
    # Initialize weights
    generator1.apply(weights_init_normal)
    generator2.apply(weights_init_normal)
    discriminator1.apply(weights_init_normal)
    discriminator2.apply(weights_init_normal)
    model.apply(weights_init_normal)

# Optimizers - Jointly train generators and STN
optimizer_G = torch.optim.Adam(itertools.chain(generator1.parameters(), generator2.parameters(), model.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(itertools.chain(discriminator1.parameters(), discriminator2.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2))

# Tensor type - only use HalfTensor in this AMP script
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
HalfTensor = torch.cuda.HalfTensor if cuda else torch.HalfTensor

##############################
# Transforms and Dataloaders
##############################

#Resizing happens in the ImageDataset() to 256 x 256 so that I can get patches (datasets_stn.py)
transforms_ = [
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

dataloader = DataLoader(
    ImageDataset(root = "./data/%s" % opt.dataset_name,
        transforms_=transforms_),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
    drop_last=True,
)

test_dataloader = DataLoader(
    TestImageDataset(root = "./data/%s" % opt.dataset_name,
        transforms_=transforms_,
        mode="test"),
    batch_size=1,
    shuffle=True,
    num_workers=1,
)

##############################
#       Training
##############################

prev_time = time.time()
f = open('./LOGS/{}.txt'.format(opt.experiment), 'a+')

# AMP
scaler = GradScaler()

for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        real_A = Variable(batch["A"].type(HalfTensor))
        real_B = Variable(batch["B"].type(HalfTensor))

        # Adversarial ground truths
        valid_ones = Variable(HalfTensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
        valid = valid_ones.fill_(0.9) # one-sided label smoothing of 0.9 vs. 1.0
        fake = Variable(HalfTensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)

        # ------------------
        #  Train Generator
        # ------------------
        optimizer_G.zero_grad()
        print("+ + + optimizer_G.zero_grad() + + + ")
        with autocast():

            # fake_B
            fake_B = generator1(real_A)
            fake_A1 = generator2(real_B)

            # pass to generator 2 for fake_A
            warped_B = model(img_A=real_A, img_B=fake_A1, src=real_B)
            fake_A2 = generator2(warped_B)  # don't detach, otherwise, the G won't get updates

            # min recon loss / you need this b/c w/o it, the visible and thermal do not look similar to reals
            recon_loss = global_pixel_loss(fake_A2, real_A)  # L1

            # perceptual loss / this is just making it look nice but ^^ recon is responsible for identity and fidelity
            perc_A = criterion_lpips(fake_A2, real_A)
            perc_B = criterion_lpips(fake_B, real_B)
            perc_loss = (perc_A + perc_B).mean()

            # triplet morph loss / all cast into same gradient modality of RGB
            morph_loss = morph_triplet(real_A, real_B, warped_B)

            # Adverarial - How fake and how real
            loss_GAN1 = global_gen_loss(real_A, real_B, fake_B, mode='B')
            loss_GAN2 = global_gen_loss(real_A, real_B, fake_A2, mode='A')
            loss_GAN = (loss_GAN1 + loss_GAN2).mean()

            # Total Loss
            alpha1 = 0.001
            alpha2 = 0.01
            loss_G = loss_GAN + alpha2*recon_loss + perc_loss + morph_loss

        scaler.scale(loss_G).backward()
        scaler.step(optimizer_G)
        print("+ + + optimizer_G.step() + + + ")

        # -----------------------
        #  Train Discriminator
        # -----------------------

        optimizer_D.zero_grad()
        print("+ + + optimizer_D.zero_grad() + + +")
        with autocast():

            loss_D1 = global_disc_loss(real_A, real_B, fake_img=fake_B, mode='B')
            loss_D2 = global_disc_loss(real_A, real_B, fake_img=fake_A2, mode='A') # fake_A2
            loss_D = 0.5*(loss_D1 + loss_D2)

        scaler.scale(loss_D).backward()
        scaler.step(optimizer_D)
        print("+ + + optimizer_D.step() + + +")

        scaler.update()

        # --------------
        #  Log Progress
        # --------------

        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, R: %f, M: %f, P: %f] ETA: %s"
            % (
                epoch, #%d
                opt.n_epochs, #%d
                i, #%d
                len(dataloader), #%d
                loss_D.item(), #%f
                loss_G.item(), #%f - total G loss
                recon_loss.item(),
                morph_loss.item(),
                perc_loss.item(),
                time_left, #%s
            )
        )

        f.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, R:%f, M: %f, P: %f] ETA: %s"
            % (
                epoch, #%d
                opt.n_epochs, #%d
                i, #%d
                len(dataloader), #%d
                loss_D.item(), #%f
                loss_G.item(), #%f - total G loss
                recon_loss.item(),
                morph_loss.item(),
                perc_loss.item(),
                time_left, #%s
            )
        )

        # If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done)


    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator1.state_dict(), "./saved_models/%s/generator1_%d.pth" % (opt.experiment, epoch))
        torch.save(generator2.state_dict(), "./saved_models/%s/generator2_%d.pth" % (opt.experiment, epoch))
        torch.save(discriminator1.state_dict(), "./saved_models/%s/discriminator1_%d.pth" % (opt.experiment, epoch))
        torch.save(discriminator2.state_dict(), "./saved_models/%s/discriminator2_%d.pth" % (opt.experiment, epoch))
        torch.save(model.state_dict(), "./saved_models/%s/stn_%d.pth" % (opt.experiment, epoch))
