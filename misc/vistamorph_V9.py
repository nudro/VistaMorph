"""
VistaMorph V9: Feature-Guided Registration with Cycle Consistency

This version implements feature-based registration to improve handling of extreme warps:
1. Added a feature extraction network to identify key points/features
2. Uses extracted features to guide the registration process
3. Maintains cycle consistency and identity preservation from V8
4. Combines feature matching with affine transformation

Key changes from V8:
- Added FeatureExtractor network for key point detection
- Added feature matching loss to guide registration
- Modified STN to incorporate feature information
- Enhanced registration process with feature guidance

The goal is to improve registration accuracy by using both pixel-level and feature-level
information, particularly for cases with extreme warps between real_A and real_B.
"""

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
from lpips import LPIPS
from kornia import morphology as morph
import kornia.contrib as K

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

######################
# Feature Extraction Network
######################

class FeatureExtractor(nn.Module):
    def __init__(self, in_channels=3):
        super(FeatureExtractor, self).__init__()
        
        # Add batch normalization to help with gradient flow
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Use sigmoid instead of softmax for keypoint detection
        self.keypoint_conv = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.descriptor_conv = nn.Conv2d(512, 256, kernel_size=1)
        
        # Use average pooling instead of max pooling
        self.pool = nn.AvgPool2d(2, 2)
        
    def forward(self, x):
        # Feature extraction
        x1 = self.conv1(x)
        x1 = self.pool(x1)
        
        x2 = self.conv2(x1)
        x2 = self.pool(x2)
        
        x3 = self.conv3(x2)
        x3 = self.pool(x3)
        
        x4 = self.conv4(x3)
        
        # Keypoint detection
        keypoints = self.keypoint_conv(x4)
        
        # Feature descriptors
        descriptors = self.descriptor_conv(x4)
        
        return keypoints, descriptors

######################
# Modified STN to incorporate feature information
######################

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(True),
            nn.Linear(in_features, in_features)
        )
    
    def forward(self, x):
        return x + self.block(x)


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
        
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        input_shape = (opt.channels, opt.img_height, opt.img_width)
        self.localization = LocalizerVIT(input_shape)
        self.feature_extractor = FeatureExtractor()
        
        # Feature fusion layer - adjusted dimensions
        self.fusion = nn.Sequential(
            nn.Linear(1*17*768 + 256*2, 1024),  # 256*2 for both feature descriptors
            nn.ReLU(True),
            nn.Linear(1024, 256),
            nn.ReLU(True),
            ResidualBlock(256),
            nn.Linear(256, 3*2),
            nn.Sigmoid()
        )
        
    def stn_phi(self, x, features_A, features_B):
        xs = self.localization(x)  # [batch, 17, 768]
        xs = xs.view(-1, 1 * xs.size(1) * xs.size(2))  # [batch, 17*768]
        
        # Concatenate features with localization output
        combined = torch.cat([xs, features_A, features_B], dim=1)  # [batch, 17*768 + 256*2]
        theta = self.fusion(combined)
        theta = theta.view(-1, 2, 3)
        return theta

    def forward(self, img_A, img_B, src):
        identity_matrix = [1, 0, 0, 0, 1, 0]

        with autocast():
            # Extract features
            keypoints_A, descriptors_A = self.feature_extractor(img_A)
            keypoints_B, descriptors_B = self.feature_extractor(img_B)
            
            # Combine features - take mean across spatial dimensions
            features_A = descriptors_A.mean(dim=[2, 3])  # [batch, 256]
            features_B = descriptors_B.mean(dim=[2, 3])  # [batch, 256]
            
            img_input = torch.cat((img_A, img_B), 1)
            dtheta = self.stn_phi(img_input, features_A, features_B)
            
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

        return torch.cat(warped), theta, keypoints_A, keypoints_B



##########################
# Loss functions
##########################
criterion_GAN = torch.nn.BCEWithLogitsLoss() # Relativistic

criterion_lpips = LPIPS(
    net='vgg',  # choose a network type from ['alex', 'squeeze', 'vgg']
    version='0.1'  # Currently, v0.1 is supported
)

criterion_L1 = nn.L1Loss()
criterion_MSE = nn.MSELoss()  # For tie point loss

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
    
    warped_B, theta, keypoints_A, keypoints_B = model(img_A=real_A, img_B=real_A, src=real_B)
    fake_A1 = generator2(warped_B)
    fake_B = generator1(fake_A1)
    
    # Visualize keypoints on images and ensure 3 channels
    keypoints_A_vis = F.interpolate(keypoints_A, size=(opt.img_height, opt.img_width), mode='bilinear', align_corners=True)
    keypoints_B_vis = F.interpolate(keypoints_B, size=(opt.img_height, opt.img_width), mode='bilinear', align_corners=True)
    
    # Expand keypoint tensors to 3 channels
    keypoints_A_vis = keypoints_A_vis.expand(-1, 3, -1, -1)
    keypoints_B_vis = keypoints_B_vis.expand(-1, 3, -1, -1)
    
    # Denormalize images
    def denormalize(x):
        return (x + 1) / 2.0
    
    real_A_denorm = denormalize(real_A.data)
    real_B_denorm = denormalize(real_B.data)
    warped_B_denorm = denormalize(warped_B.data)
    fake_A1_denorm = denormalize(fake_A1.data)
    fake_B_denorm = denormalize(fake_B.data)
    
    # Stack images
    img_sample_global = torch.cat((
        real_A_denorm, real_B_denorm, warped_B_denorm, 
        fake_A1_denorm, fake_B_denorm,
        keypoints_A_vis.data, keypoints_B_vis.data
    ), -1)
    
    # Save with proper normalization
    save_image(img_sample_global, "./images/%s/%s.png" % (opt.experiment, batches_done), nrow=4, normalize=False)


##########################
# GENERATORS
##########################

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

#################
# LOSSES
#################

def feature_matching_loss(keypoints_A, keypoints_B, theta):
    batch_size = keypoints_A.size(0)
    loss = 0
    
    for b in range(batch_size):
        # Get top K keypoints with non-maximum suppression
        kp_A = keypoints_A[b].view(-1)
        kp_B = keypoints_B[b].view(-1)
        
        # Apply non-maximum suppression
        scores_A, idx_A = nms(kp_A)
        scores_B, idx_B = nms(kp_B)
        
        # Convert to 2D coordinates with proper scaling
        h, w = keypoints_A.size(2), keypoints_A.size(3)
        scale_h = opt.img_height / h
        scale_w = opt.img_width / w
        
        y_A = (idx_A // w).float() * scale_h
        x_A = (idx_A % w).float() * scale_w
        y_B = (idx_B // w).float() * scale_h
        x_B = (idx_B % w).float() * scale_w
        
        # Stack coordinates
        coords_A = torch.stack([x_A, y_A], dim=1)
        coords_B = torch.stack([x_B, y_B], dim=1)
        
        # Apply transformation to A coordinates with better numerical stability
        theta_b = theta[b].view(2, 3)
        ones = torch.ones(coords_A.size(0), 1).cuda()
        coords_A_homo = torch.cat([coords_A, ones], dim=1)
        transformed_A = torch.mm(coords_A_homo, theta_b.t())
        
        # Calculate distance with L2 norm and add epsilon for numerical stability
        dist = torch.cdist(transformed_A, coords_B)
        min_dist, _ = torch.min(dist, dim=1)
        
        # Add confidence weighting with better scaling
        confidence = torch.clamp(scores_A * scores_B, min=1e-6)
        weighted_loss = (min_dist * confidence).mean()
        
        # Use a more stable normalization
        normalized_loss = weighted_loss / (torch.sqrt(torch.tensor(opt.img_height * opt.img_width).float()))
        loss += normalized_loss
    
    return loss / batch_size
    
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

##############################
# Transforms and Dataloaders
##############################

#Resizing happens in the ImageDataset() to 256 x 256 so that I can get patches (datasets_stn.py)
transforms_ = [
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize to [-1, 1]
]

dataloader = DataLoader(
    ImageDataset(root = "./data/%s" % opt.dataset_name,
        transforms_=transforms_,
        mode="train"),
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
    #FeatureExtractor = FeatureExtractor.cuda()

    criterion_GAN.cuda()
    criterion_lpips.cuda()
    criterion_L1.cuda()
    criterion_MSE.cuda()

# Trained on multigpus - change your device ids if needed
generator1 = torch.nn.DataParallel(generator1, device_ids=[0,1])
generator2 = torch.nn.DataParallel(generator2, device_ids=[0,1])
discriminator1 = torch.nn.DataParallel(discriminator1, device_ids=[0,1])
discriminator2 = torch.nn.DataParallel(discriminator2, device_ids=[0,1])
model = torch.nn.DataParallel(model, device_ids=[0,1])
#FeatureExtractor = torch.nn.DataParallel(model, device_ids=[0,1])


if opt.epoch != 0:
    # Load pretrained models
    generator1.load_state_dict(torch.load("./saved_models/%s/generator1_%d.pth" % (opt.experiment, opt.epoch)))
    generator2.load_state_dict(torch.load("./saved_models/%s/generator2_%d.pth" % (opt.experiment, opt.epoch)))
    discriminator1.load_state_dict(torch.load("./saved_models/%s/discriminator1_%d.pth" % (opt.experiment, opt.epoch)))
    discriminator2.load_state_dict(torch.load("./saved_models/%s/discriminator2_%d.pth" % (opt.experiment, opt.epoch)))
    model.load_state_dict(torch.load("./saved_models/%s/stn_%d.pth" % (opt.experiment, opt.epoch)))
    #FeatureExtractor.load_state_dict(torch.load("./saved_models/%s/fe_%d.pth" % (opt.experiment, opt.epoch)))

else:
    # Initialize weights
    generator1.apply(weights_init_normal)
    generator2.apply(weights_init_normal)
    discriminator1.apply(weights_init_normal)
    discriminator2.apply(weights_init_normal)
    model.apply(weights_init_normal)
    #FeatureExtractor.apply(weights_init_normal)

# Optimizers - Jointly train generators and STN
optimizer_G = torch.optim.Adam(itertools.chain(generator1.parameters(), generator2.parameters(), model.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(itertools.chain(discriminator1.parameters(), discriminator2.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2))

# Tensor type - only use HalfTensor in this AMP script
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
HalfTensor = torch.cuda.HalfTensor if cuda else torch.HalfTensor

##############################
#       Training
##############################

prev_time = time.time()
f = open('./LOGS/{}.txt'.format(opt.experiment), 'a+')

# AMP
scaler = GradScaler()

# Add these constants at the top of the file after the imports
LAMBDA_FEATURE = 0.1  # Weight for feature matching loss
MAX_GRAD_NORM = 1.0   # Maximum gradient norm for clipping

for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        real_A = Variable(batch["A"].type(HalfTensor))
        real_B = Variable(batch["B"].type(HalfTensor))
        Y = Variable(batch["Y"].type(HalfTensor))  # Ground truth affine matrix

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
            # Spatial transformation to align real_B with real_A
            warped_B, theta, keypoints_A, keypoints_B = model(img_A=real_A, img_B=real_A, src=real_B)
            
            # Generate fake_A1 from warped_B - this should maintain real_A features
            fake_A1 = generator2(warped_B)
            fake_B = generator1(fake_A1)
            
            # Add cycle consistency to ensure reversible transformations
            cycle_A = generator2(fake_B)  # Should reconstruct real_A
            cycle_B = generator1(fake_A1)  # Should reconstruct warped_B
            
            # Reconstruction losses
            recon_loss1 = global_pixel_loss(fake_A1, real_A)
            recon_loss2 = global_pixel_loss(fake_B, warped_B)
            recon_loss = (recon_loss1 + recon_loss2).mean()
            
            # Cycle consistency loss to ensure transformations are reversible
            cycle_loss = (global_pixel_loss(cycle_A, real_A) + global_pixel_loss(cycle_B, warped_B)).mean()
            
            # Identity loss to maintain real_A features in fake_A1
            identity_loss = global_pixel_loss(fake_A1, real_A)

            # Perceptual loss using LPIPS
            perc_A = criterion_lpips(fake_A1, real_A)
            perc_B = criterion_lpips(fake_B, warped_B)
            perc_loss = (perc_A + perc_B).mean()

            # Adversarial losses
            loss_GAN1 = global_gen_loss(real_A, warped_B, fake_B, mode='B')
            loss_GAN2 = global_gen_loss(real_A, warped_B, fake_A1, mode='A')
            loss_GAN = (loss_GAN1 + loss_GAN2).mean()

            # Tie point loss - MSE between predicted and ground truth affine matrices
            tie_loss = criterion_MSE(theta, Y)

            # Feature matching loss
            feat_loss = feature_matching_loss(keypoints_A, keypoints_B, theta)

            # Total Loss with adjusted weights
            alpha1 = 0.05  # Cycle consistency weight
            alpha2 = 0.25  # Tie point loss weight
            alpha3 = 0.1   # Identity loss weight
            alpha4 = 0.15  # Feature matching weight
            loss_G = (loss_GAN + recon_loss + perc_loss + 
                     alpha1*cycle_loss + alpha2*tie_loss + alpha3*identity_loss +
                     alpha4*feat_loss).mean()

        # Compute feature matching loss
        feature_loss = feature_matching_loss(keypoints_A, keypoints_B, theta)
        
        # Combine losses with weighting
        total_loss = loss_G + LAMBDA_FEATURE * feature_loss
        
        # Backward pass with gradient clipping
        scaler.scale(total_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        scaler.step(optimizer_G)
        print("+ + + optimizer_G.step() + + + ")

        # -----------------------
        #  Train Discriminator
        # -----------------------

        optimizer_D.zero_grad()
        print("+ + + optimizer_D.zero_grad() + + +")
        with autocast():
            loss_D1 = global_disc_loss(real_A, real_B, fake_img=fake_B, mode='B')
            loss_D2 = global_disc_loss(real_A, real_B, fake_img=fake_A1, mode='A')
            loss_D = (loss_D1 + loss_D2).mean()

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
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, R: %f, P: %f, T: %f, F: %f] ETA: %s"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(dataloader),
                loss_D.item(),
                loss_G.item(),
                recon_loss.item(),
                perc_loss.item(),
                tie_loss.item()*0.25,
                feat_loss.item(),
                time_left,
            )
        )

        f.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, R:%f, P: %f, T: %f, F: %f] ETA: %s"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(dataloader),
                loss_D.item(),
                loss_G.item(),
                recon_loss.item(),
                perc_loss.item(),
                tie_loss.item(),
                feat_loss.item(),
                time_left,
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
        torch.save(model.state_dict(), "./saved_models/%s/net_%d.pth" % (opt.experiment, epoch)) 
        #torch.save(FeatureExtractor.state_dict(), "./saved_models/%s/fe_%d.pth" % (opt.experiment, epoch)) 

        