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
from datasets_stn_with_labels import * # only A and B
import kornia
import kornia.feature as KF
from lpips import LPIPS
from kornia import morphology as morph
import kornia.contrib as K
import matplotlib.pyplot as plt


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
parser.add_argument("--checkpoint_interval", type=int, default=20, help="interval between model checkpoints")
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
criterion_GAN = torch.nn.BCEWithLogitsLoss() # Relativitic

criterion_lpips = LPIPS(
    net='vgg',  # choose a network type from ['alex', 'squeeze', 'vgg']
    version='0.1'  # Currently, v0.1 is supported
)

criterion_L1 = nn.L1Loss()



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


def visualize_tiepoints(img1, img2, source_points, target_points, title, save_path):
    """
    Visualize tiepoints between two images.
    Args:
        img1: First image tensor [1, 3, H, W]
        img2: Second image tensor [1, 3, H, W]
        source_points: Source points tensor [4, 2]
        target_points: Target points tensor [4, 2]
        title: Title for the plot
        save_path: Path to save the visualization
    """
    # Convert tensors to numpy arrays
    img1_np = img1.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    img2_np = img2.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    
    # Denormalize images
    img1_np = (img1_np * 0.5 + 0.5) * 255
    img2_np = (img2_np * 0.5 + 0.5) * 255
    
    # Convert points to numpy arrays
    source_points = source_points.squeeze(0).cpu().numpy()
    target_points = target_points.squeeze(0).cpu().numpy()
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot images side by side
    plt.subplot(121)
    plt.imshow(img1_np.astype(np.uint8))
    plt.title('Source Image')
    plt.axis('off')
    
    plt.subplot(122)
    plt.imshow(img2_np.astype(np.uint8))
    plt.title('Target Image')
    plt.axis('off')
    
    # Plot tiepoints and connections
    colors = ['red', 'green', 'blue', 'yellow']
    for i in range(4):
        # Plot points
        plt.subplot(121)
        plt.scatter(source_points[i, 0], source_points[i, 1], c=colors[i], marker='o', s=100)
        
        plt.subplot(122)
        plt.scatter(target_points[i, 0], target_points[i, 1], c=colors[i], marker='o', s=100)
    
    plt.suptitle(title)
    plt.savefig(save_path)
    plt.close()

def visualize_loftr_features(img1, img2, title, save_path):
    """
    Visualize LoFTR features between two images.
    Args:
        img1: First image tensor [1, 3, H, W]
        img2: Second image tensor [1, 3, H, W]
        title: Title for the plot
        save_path: Path to save the visualization
    Returns:
        features: Dictionary containing LoFTR features
    """
    # Convert tensors to numpy arrays and denormalize
    img1_np = img1.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
    img2_np = img2.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
    
    img1_np = (img1_np * 0.5 + 0.5) * 255
    img2_np = (img2_np * 0.5 + 0.5) * 255
    
    # Convert to grayscale for LoFTR and ensure FloatTensor
    img1_gray = kornia.color.rgb_to_grayscale(img1).type(torch.cuda.FloatTensor)
    img2_gray = kornia.color.rgb_to_grayscale(img2).type(torch.cuda.FloatTensor)
    
    # Initialize LoFTR matcher and move to GPU
    matcher = KF.LoFTR(pretrained='outdoor').cuda()
    
    # Get correspondences
    input_dict = {
        "image0": img1_gray,
        "image1": img2_gray
    }
    
    with torch.no_grad():
        features = matcher(input_dict)
    
    # Convert keypoints to numpy
    mkpts0 = features['keypoints0'].detach().cpu().numpy()
    mkpts1 = features['keypoints1'].detach().cpu().numpy()
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    
    # Plot images side by side
    plt.subplot(121)
    plt.imshow(img1_np.astype(np.uint8))
    plt.title('Source Image')
    plt.axis('off')
    
    plt.subplot(122)
    plt.imshow(img2_np.astype(np.uint8))
    plt.title('Target Image')
    plt.axis('off')
    
    # Plot keypoints and connections
    plt.subplot(121)
    plt.scatter(mkpts0[:, 0], mkpts0[:, 1], c='red', marker='.', s=1)
    
    plt.subplot(122)
    plt.scatter(mkpts1[:, 0], mkpts1[:, 1], c='red', marker='.', s=1)
    
    # Draw lines between corresponding points
    for i in range(len(mkpts0)):
        plt.plot([mkpts0[i, 0], mkpts1[i, 0]], 
                [mkpts0[i, 1], mkpts1[i, 1]], 
                'c-', alpha=0.3, linewidth=0.5)
    
    plt.suptitle(title)
    plt.savefig(save_path)
    plt.close()
    
    return features

def sample_images(batches_done):
    imgs = next(iter(test_dataloader)) # batch_size = 1
    real_A = Variable(imgs["A"].type(HalfTensor)) # torch.Size([1, 3, 256, 256])
    real_B = Variable(imgs["B"].type(HalfTensor))
    
    fake_A = generator2(real_B) # only using generator1

    # pass to generator 2 for fake_A
    warped_B, theta = model(img_A=real_A, img_B=fake_A, src=real_B)
    fake_A2 = generator2(warped_B)

    # Get LoFTR features and visualize for original pair
    original_features = visualize_loftr_features(
        real_A, real_B,
        'Original Pair LoFTR Features',
        f"./images/{opt.experiment}/loftr_original_{batches_done}.png"
    )
    
    # Get LoFTR features and visualize for warped pair
    warped_features = visualize_loftr_features(
        real_A, warped_B,
        'Warped Pair LoFTR Features',
        f"./images/{opt.experiment}/loftr_warped_{batches_done}.png"
    )
    
    # Print feature statistics
    print(f"Original pair features: {len(original_features['keypoints0'])} matches")
    print(f"Warped pair features: {len(warped_features['keypoints0'])} matches")
    
    img_sample_global = torch.cat((real_A.data, real_B.data, fake_A2.data, warped_B.data, fake_A.data), -1)
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

        self.fc_loc = nn.Sequential(
            nn.Linear(1*17*768, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 3*2),
            nn.Sigmoid())
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
            # Expand edge-detected images to 3 channels each
            edge_A_expanded = img_A.expand(-1, 3, -1, -1)  # [batch, 3, H, W]
            edge_B_expanded = img_B.expand(-1, 3, -1, -1)  # [batch, 3, H, W]
            
            # Concatenate expanded edge images
            img_input = torch.cat((edge_A_expanded, edge_B_expanded), 1)  # [batch, 6, H, W]
            dtheta = self.stn_phi(img_input)
            
            identity_theta = torch.tensor(identity_matrix, dtype=torch.float).cuda()
            dtheta = dtheta.reshape(img_A.size(0), 2*3)
            theta = dtheta + identity_theta.unsqueeze(0).repeat(img_A.size(0),1)

            # Apply transformation
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
                Rs = F.grid_sample(src_tensors[i], rs_grid, mode='bicubic', padding_mode='border', align_corners=True)
                warped.append(Rs.type(HalfTensor))

        return torch.cat(warped), theta


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

def global_pixel_loss(fake_A, real_A):
    loss_pix = criterion_L1(fake_A, real_A) # LPIPS
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
        loss_D = (loss_real + loss_fake).mean()

    if mode=='A':
        pred_real = discriminator2(real_A, real_B)
        pred_fake = discriminator2(fake_img.detach(), real_B)

        loss_real = criterion_GAN(pred_real - pred_fake, valid)
        loss_fake = criterion_GAN(pred_fake - pred_real, fake)
        loss_D = (loss_real + loss_fake).mean()

    return loss_D

def loftr_feature_loss(img1, img2):
    """
    Calculate LoFTR feature loss between two images.
    Combines:
    1. Mean Euclidean Distance (MED)
    2. Symmetric Transfer Error
    3. Confidence-weighted Error
    Args:
        img1: First image tensor [1, 3, H, W]
        img2: Second image tensor [1, 3, H, W]
    Returns:
        loss: Combined LoFTR feature loss
    """
    # Convert to grayscale for LoFTR and ensure FloatTensor
    img1_gray = kornia.color.rgb_to_grayscale(img1).type(torch.cuda.FloatTensor)
    img2_gray = kornia.color.rgb_to_grayscale(img2).type(torch.cuda.FloatTensor)
    
    # Initialize LoFTR matcher and move to GPU
    matcher = KF.LoFTR(pretrained='outdoor').cuda()
    
    # Get correspondences in both directions
    input_dict_forward = {
        "image0": img1_gray,
        "image1": img2_gray
    }
    input_dict_backward = {
        "image0": img2_gray,
        "image1": img1_gray
    }
    
    with torch.no_grad():
        features_forward = matcher(input_dict_forward)
        features_backward = matcher(input_dict_backward)
    
    # Get keypoints and confidence scores
    mkpts0_forward = features_forward['keypoints0']
    mkpts1_forward = features_forward['keypoints1']
    conf_forward = features_forward['confidence']
    
    mkpts0_backward = features_backward['keypoints0']
    mkpts1_backward = features_backward['keypoints1']
    conf_backward = features_backward['confidence']
    
    # 1. Mean Euclidean Distance (MED)
    dist_forward = torch.norm(mkpts0_forward - mkpts1_forward, dim=1)
    dist_backward = torch.norm(mkpts0_backward - mkpts1_backward, dim=1)
    med_loss = (dist_forward.mean() + dist_backward.mean()) / 2
    
    # 2. Symmetric Transfer Error
    # Find corresponding points in both directions
    # This is a simplified version - in practice, you'd want to do proper point matching
    sym_error = torch.abs(dist_forward - dist_backward).mean()
    
    # 3. Confidence-weighted Error
    conf_weighted_forward = (conf_forward * dist_forward).sum() / conf_forward.sum()
    conf_weighted_backward = (conf_backward * dist_backward).sum() / conf_backward.sum()
    conf_weighted_loss = (conf_weighted_forward + conf_weighted_backward) / 2
    
    # Combine losses with weights
    total_loss = med_loss + 0.5 * sym_error + 0.5 * conf_weighted_loss
    
    return total_loss

# ===========================================================
# Initialize generator and discriminator
# ===========================================================

# Gen1/Disc1 == B, Gen2/Disc2 = A
input_shape_patch = (opt.channels, opt.img_height, opt.img_width)
generator2 = GeneratorUNet2(input_shape_patch) # for fake_A
discriminator2 = Discriminator2(input_shape_patch) # for fake_A

if cuda:
    generator2 = generator2.cuda()
    discriminator2 = discriminator2.cuda()
    model = Net().cuda()

    criterion_GAN.cuda()
    criterion_L1.cuda()
    criterion_lpips.cuda()
    #criterion_amp.cuda()
    #criterion_phase.cuda()

# Trained on multigpus - change your device ids if needed
generator2 = torch.nn.DataParallel(generator2, device_ids=[0, 1])
discriminator2 = torch.nn.DataParallel(discriminator2, device_ids=[0, 1])
model = torch.nn.DataParallel(model, device_ids=[0, 1])

if opt.epoch != 0:
    # Load pretrained models
    generator2.load_state_dict(torch.load("./saved_models/%s/generator2_%d.pth" % (opt.experiment, opt.epoch)))
    discriminator2.load_state_dict(torch.load("./saved_models/%s/discriminator2_%d.pth" % (opt.experiment, opt.epoch)))
    model.load_state_dict(torch.load("./saved_models/%s/stn_%d.pth" % (opt.experiment, opt.epoch)))

else:
    # Initialize weights
    generator2.apply(weights_init_normal)
    discriminator2.apply(weights_init_normal)
    model.apply(weights_init_normal)

# Optimizers - Jointly train generators and STN

optimizer_G = torch.optim.Adam(itertools.chain(generator2.parameters(),  model.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator2.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Tensor type - only use HalfTensor in this AMP script
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
HalfTensor = torch.cuda.HalfTensor if cuda else torch.HalfTensor

########################
# FUNCTINOS
########################
def affine_to_tiepoints(theta, img_size=(256, 256)):
    """
    Convert affine transformation matrix to tiepoints.
    Args:
        theta: Affine transformation matrix [B, 2, 3] or [2, 3]
        img_size: Tuple of (height, width) of the image
    Returns:
        source_points: Source points in the original image [B, 4, 2]
        target_points: Target points after transformation [B, 4, 2]
    """
    if len(theta.shape) == 2:
        theta = theta.unsqueeze(0)  # Add batch dimension if not present
    
    batch_size = theta.shape[0]
    h, w = img_size
    
    # Define corner points of the image
    corners = torch.tensor([
        [0, 0],      # top-left
        [w-1, 0],    # top-right
        [w-1, h-1],  # bottom-right
        [0, h-1]     # bottom-left
    ], dtype=theta.dtype, device=theta.device)
    
    # Expand corners for batch processing
    source_points = corners.unsqueeze(0).repeat(batch_size, 1, 1)
    
    # Convert corners to homogeneous coordinates
    ones = torch.ones((batch_size, 4, 1), dtype=theta.dtype, device=theta.device)
    source_points_h = torch.cat([source_points, ones], dim=2)
    
    # Apply transformation
    target_points = torch.bmm(source_points_h, theta.transpose(1, 2))
    
    return source_points, target_points


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
        Y = Variable(batch["Y"].type(HalfTensor))  # Ground truth affine matrix

        # Adversarial ground truths
        valid_ones = Variable(HalfTensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
        valid = valid_ones.fill_(0.9)
        fake = Variable(HalfTensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)

        # ------------------
        #  Train Generator
        # ------------------
        optimizer_G.zero_grad()
        print("+ + + optimizer_G.zero_grad() + + + ")
        with autocast():

            # fake_B
            fake_A = generator2(real_B) # only using generator2 -- will learn same features of fake_'aness

            # pass to generator 2 for fake_A
            warped_B, theta = model(img_A=fake_A, img_B=real_A, src=real_B)
            fake_A2 = generator2(warped_B)

            # min recon loss
            recon_loss = global_pixel_loss(fake_A2, real_A)

            # perceptual loss, LPIPS
            perc_A = criterion_lpips(fake_A2, real_A)
            perc_loss = perc_A.mean()

            # Replace tie_error with LoFTR feature loss
            loftr_loss = loftr_feature_loss(warped_B, real_A)

            # Adverarial ~ mode A goes to Discirminator 2
            loss_GAN = global_gen_loss(real_A, warped_B, fake_A2, mode='A') # need registered B <> real_A and registered B <> fake_A2 identifcal


            # Total Loss
            alpha1 = 0.001
            alpha2 = 0.25
            loss_G = (loss_GAN + loftr_loss + perc_loss + recon_loss).mean()

        scaler.scale(loss_G).backward()
        scaler.step(optimizer_G)
        print("+ + + optimizer_G.step() + + + ")

        # -----------------------
        #  Train Discriminator
        # -----------------------

        optimizer_D.zero_grad()
        print("+ + + optimizer_D.zero_grad() + + +")
        with autocast():
            loss_D2 = global_disc_loss(real_A, warped_B.detach(), fake_img=fake_A2, mode='A') #only using d2
            loss_D = loss_D2

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
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, R(L1): %f, Perc(L1): %f, LoFTR: %f] ETA: %s"
            % (
                epoch, #%d
                opt.n_epochs, #%d
                i, #%d
                len(dataloader), #%d
                loss_D.item(), #%f
                loss_G.item(), #%f - total G loss
                recon_loss.item(),
                perc_loss.item(),
                loftr_loss.item(),
                time_left, #%s
            )
        )

        f.write(
           "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, R(L1): %f, Perc(L1): %f, LoFTR: %f] ETA: %s"
            % (
                epoch, #%d
                opt.n_epochs, #%d
                i, #%d
                len(dataloader), #%d
                loss_D.item(), #%f
                loss_G.item(), #%f - total G loss
                recon_loss.item(),
                perc_loss.item(),
                loftr_loss.item(),
                time_left, #%s
            )
        )

        # If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done)


    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator2.state_dict(), "./saved_models/%s/generator2_%d.pth" % (opt.experiment, epoch))
        torch.save(discriminator2.state_dict(), "./saved_models/%s/discriminator2_%d.pth" % (opt.experiment, epoch))
        torch.save(model.state_dict(), "./saved_models/%s/stn_%d.pth" % (opt.experiment, epoch))
