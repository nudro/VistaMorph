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
import antialiased_cnns
from datasets_stn import * # only A and B
import kornia
from lpips import LPIPS
from kornia import morphology as morph
import kornia.contrib as K


"""
Apple Silicon (M3) version of VISTA-MORPH
Modified to use MPS (Metal Performance Shaders) instead of CUDA
"""

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=210, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="Carl_Final", help="name of the dataset")
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

os.makedirs("images/%s" % opt.experiment, exist_ok=True)
os.makedirs("saved_models/%s" % opt.experiment, exist_ok=True)
os.makedirs("LOGS/%s" % opt.experiment, exist_ok=True)

# Check if MPS is available
mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
device = torch.device("mps" if mps_available else "cpu")
print(f"Using device: {device}")

torch.manual_seed(42)

##########################
# Loss functions
##########################
criterion_GAN = torch.nn.BCEWithLogitsLoss()
criterion_lpips = LPIPS().to(device)
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
    t = tensor.to(device)
    t = t.reshape(t.size(0), 1, t.size(1), t.size(2))
    t = t.expand(-1, 3, -1, -1) # needs to be [1, 3, 256, 256]
    return t


def sample_images(batches_done):
    imgs = next(iter(test_dataloader))
    real_A = Variable(imgs["A"].type(torch.float32)).to(device)
    real_B = Variable(imgs["B"].type(torch.float32)).to(device)
    fake_B = generator1(real_A)
    fake_A1 = generator2(real_B)
    warped_B = model(img_A=real_A, img_B=fake_A1, src=real_B)
    img_sample_global = torch.cat((real_A.data, real_B.data, warped_B.data, fake_A1.data, fake_B.data), -1)
    save_image(img_sample_global, "images/%s/%s.png" % (opt.experiment, batches_done), nrow=4, normalize=True)

#################
# ViT for STN
################


class LocalizerVIT(nn.Module):
    def __init__(self, img_shape):
        super(LocalizerVIT, self).__init__()
        channels, self.h, self.w = img_shape
        self.vit = nn.Sequential(
            K.VisionTransformer(
                image_size=self.h,
                patch_size=16,
                in_channels=channels*2
            )
        )

    def forward(self, x):
        out = self.vit(x)
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
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
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
        # U-Net generator with skip connections from encoder to decoder
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

        return self.final(u5)


class GeneratorUNet2(nn.Module):
    def __init__(self, img_shape):
        super(GeneratorUNet2, self).__init__()
        channels, self.h, self.w = img_shape

        self.down1 = UNetDown(channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
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
        # U-Net generator with skip connections from encoder to decoder
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

        return self.final(u5)


class Discriminator1(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator1, self).__init__()
        channels, height, width = img_shape

        def discriminator_block(in_filters, out_filters, stride=2, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=stride, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=False))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels * 2, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)


class Discriminator2(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator2, self).__init__()
        channels, height, width = img_shape

        def discriminator_block(in_filters, out_filters, stride=2, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=stride, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=False))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels * 2, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)


def morph_triplet(real_A, real_B, reg_B):
    # Morphological triplet loss
    # Anchor: real_A
    # Positive: real_B
    # Negative: reg_B
    # Morphological gradient
    kernel = torch.ones(3, 3).to(device)
    anchor = morph.gradient(real_A, kernel)
    positive = morph.gradient(real_B, kernel)
    negative = morph.gradient(reg_B, kernel)
    return criterion_morph(anchor, positive, negative)


def global_pixel_loss(real_B, fake_B):
    return criterion_L1(real_B, fake_B)


def global_gen_loss(real_A, real_B, fake_img, mode):
    if mode == 'B':
        pred_real = discriminator1(real_A, real_B)
        pred_fake = discriminator1(real_A, fake_img)
    else:
        pred_real = discriminator2(real_A, real_B)
        pred_fake = discriminator2(real_A, fake_img)
    return -torch.mean(torch.abs(pred_real - torch.mean(pred_fake)) + torch.abs(pred_fake - torch.mean(pred_real)))


def global_disc_loss(real_A, real_B, fake_img, mode):
    if mode == 'B':
        pred_real = discriminator1(real_A, real_B)
        pred_fake = discriminator1(real_A, fake_img)
    else:
        pred_real = discriminator2(real_A, real_B)
        pred_fake = discriminator2(real_A, fake_img)
    return torch.mean(torch.abs(pred_real - torch.mean(pred_fake)) + torch.abs(pred_fake - torch.mean(pred_real)))

if __name__ == '__main__':
    # Initialize models and move to device
    input_shape_patch = (opt.channels, opt.img_height, opt.img_width)
    generator1 = GeneratorUNet1(input_shape_patch).to(device)
    generator2 = GeneratorUNet2(input_shape_patch).to(device)
    discriminator1 = Discriminator1(input_shape_patch).to(device)
    discriminator2 = Discriminator2(input_shape_patch).to(device)
    model = Net().to(device)

    criterion_GAN = criterion_GAN.to(device)
    criterion_lpips = criterion_lpips.to(device)
    criterion_L1 = criterion_L1.to(device)
    criterion_morph = criterion_morph.to(device)

    if opt.epoch != 0:
        # Load pretrained models
        generator1.load_state_dict(torch.load("saved_models/%s/generator1_%d.pth" % (opt.experiment, opt.epoch), map_location=device))
        generator2.load_state_dict(torch.load("saved_models/%s/generator2_%d.pth" % (opt.experiment, opt.epoch), map_location=device))
        discriminator1.load_state_dict(torch.load("saved_models/%s/discriminator1_%d.pth" % (opt.experiment, opt.epoch), map_location=device))
        discriminator2.load_state_dict(torch.load("saved_models/%s/discriminator2_%d.pth" % (opt.experiment, opt.epoch), map_location=device))
        model.load_state_dict(torch.load("saved_models/%s/stn_%d.pth" % (opt.experiment, opt.epoch), map_location=device))
    else:
        # Initialize weights
        generator1.apply(weights_init_normal)
        generator2.apply(weights_init_normal)
        discriminator1.apply(weights_init_normal)
        discriminator2.apply(weights_init_normal)
        model.apply(weights_init_normal)

    # Optimizers
    optimizer_G = torch.optim.Adam(
        itertools.chain(generator1.parameters(), generator2.parameters(), model.parameters()),
        lr=opt.lr,
        betas=(opt.b1, opt.b2)
    )
    optimizer_D = torch.optim.Adam(
        itertools.chain(discriminator1.parameters(), discriminator2.parameters()),
        lr=opt.lr,
        betas=(opt.b1, opt.b2)
    )

    ##############################
    # Transforms and Dataloaders
    ##############################

    transforms_ = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    dataloader = DataLoader(
        ImageDataset(root="data/%s" % opt.dataset_name, transforms_=transforms_),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for debugging
        drop_last=True,
    )

    test_dataloader = DataLoader(
        TestImageDataset(root="data/%s" % opt.dataset_name, transforms_=transforms_, mode="test"),
        batch_size=1,
        shuffle=True,
        num_workers=0,  # Set to 0 for debugging
    )

    ##############################
    #       Training
    ##############################

    prev_time = time.time()
    f = open('LOGS/{}.txt'.format(opt.experiment), 'a+')

    for epoch in range(opt.epoch, opt.n_epochs):
        for i, batch in enumerate(dataloader):
            # Configure input
            real_A = Variable(batch["A"].type(torch.float32)).to(device)
            real_B = Variable(batch["B"].type(torch.float32)).to(device)

            # Adversarial ground truths
            valid = Variable(torch.ones((real_A.size(0), *patch)), requires_grad=False).to(device)
            valid = valid.fill_(0.9)  # one-sided label smoothing
            fake = Variable(torch.zeros((real_A.size(0), *patch)), requires_grad=False).to(device)

            # ------------------
            #  Train Generator
            # ------------------
            optimizer_G.zero_grad()
            print("+ + + optimizer_G.zero_grad() + + + ")

            # Generate fake samples
            fake_B = generator1(real_A)
            fake_A1 = generator2(real_B)
            warped_B = model(img_A=real_A, img_B=fake_A1, src=real_B)
            fake_A2 = generator2(warped_B)

            # Reconstruction loss
            recon_loss = criterion_L1(fake_A2, real_A)

            # Perceptual loss
            perc_A = criterion_lpips(fake_A2, real_A)
            perc_B = criterion_lpips(fake_B, real_B)
            perc_loss = (perc_A + perc_B).mean()

            # Triplet morph loss
            morph_loss = morph_triplet(real_A, real_B, warped_B)

            # Adversarial loss
            loss_GAN1 = global_gen_loss(real_A, real_B, fake_B, mode='B')
            loss_GAN2 = global_gen_loss(real_A, real_B, fake_A2, mode='A')
            loss_GAN = (loss_GAN1 + loss_GAN2).mean()

            # Total loss
            alpha1 = 0.001
            alpha2 = 0.01
            loss_G = loss_GAN + alpha2*recon_loss + perc_loss + morph_loss

            loss_G.backward(retain_graph=True)
            optimizer_G.step()
            print("+ + + optimizer_G.step() + + + ")

            # -----------------------
            #  Train Discriminator
            # -----------------------
            optimizer_D.zero_grad()
            print("+ + + optimizer_D.zero_grad() + + +")

            # Move tensors to CPU for discriminator training
            real_A_cpu = real_A.cpu()
            real_B_cpu = real_B.cpu()
            fake_B_cpu = fake_B.detach().cpu()
            fake_A2_cpu = fake_A2.detach().cpu()

            # Move discriminators to CPU
            discriminator1.cpu()
            discriminator2.cpu()

            loss_D1 = global_disc_loss(real_A_cpu, real_B_cpu, fake_B_cpu, mode='B')
            loss_D2 = global_disc_loss(real_A_cpu, real_B_cpu, fake_A2_cpu, mode='A')
            loss_D = 0.5 * (loss_D1 + loss_D2)

            loss_D.backward()
            optimizer_D.step()
            print("+ + + optimizer_D.step() + + +")

            # Move discriminators back to MPS
            discriminator1.to(device)
            discriminator2.to(device)

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
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    loss_D.item(),
                    loss_G.item(),
                    recon_loss.item(),
                    morph_loss.item(),
                    perc_loss.item(),
                    time_left,
                )
            )

            f.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, R:%f, M: %f, P: %f] ETA: %s"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    loss_D.item(),
                    loss_G.item(),
                    recon_loss.item(),
                    morph_loss.item(),
                    perc_loss.item(),
                    time_left,
                )
            )

            if batches_done % opt.sample_interval == 0:
                sample_images(batches_done)

        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(generator1.state_dict(), "saved_models/%s/generator1_%d.pth" % (opt.experiment, epoch))
            torch.save(generator2.state_dict(), "saved_models/%s/generator2_%d.pth" % (opt.experiment, epoch))
            torch.save(discriminator1.state_dict(), "saved_models/%s/discriminator1_%d.pth" % (opt.experiment, epoch))
            torch.save(discriminator2.state_dict(), "saved_models/%s/discriminator2_%d.pth" % (opt.experiment, epoch))
            torch.save(model.state_dict(), "saved_models/%s/stn_%d.pth" % (opt.experiment, epoch)) 