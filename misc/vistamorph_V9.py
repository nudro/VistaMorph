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

# Feature Extraction Network
class FeatureExtractor(nn.Module):
    def __init__(self, in_channels=3):
        super(FeatureExtractor, self).__init__()
        
        # Feature extraction layers
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        
        # Feature point detection
        self.keypoint_conv = nn.Conv2d(512, 1, kernel_size=1)
        
        # Feature descriptor
        self.descriptor_conv = nn.Conv2d(512, 256, kernel_size=1)
        
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x):
        # Feature extraction
        x1 = self.relu(self.conv1(x))
        x1 = self.pool(x1)
        
        x2 = self.relu(self.conv2(x1))
        x2 = self.pool(x2)
        
        x3 = self.relu(self.conv3(x2))
        x3 = self.pool(x3)
        
        x4 = self.relu(self.conv4(x3))
        
        # Keypoint detection
        keypoints = self.keypoint_conv(x4)
        keypoints = F.softmax(keypoints.view(keypoints.size(0), -1), dim=1)
        keypoints = keypoints.view(keypoints.size(0), 1, x4.size(2), x4.size(3))
        
        # Feature descriptors
        descriptors = self.descriptor_conv(x4)
        
        return keypoints, descriptors

# Modified STN to incorporate feature information
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        input_shape = (opt.channels, opt.img_height, opt.img_width)
        self.localization = LocalizerVIT(input_shape)
        self.feature_extractor = FeatureExtractor()
        
        # Feature fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(1*17*768 + 256, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 256),
            nn.ReLU(True),
            ResidualBlock(256),
            nn.Linear(256, 3*2),
            nn.Sigmoid()
        )
        
    def stn_phi(self, x, features_A, features_B):
        xs = self.localization(x)
        xs = xs.view(-1, 1 * xs.size(1) * xs.size(2))
        
        # Concatenate features with localization output
        combined = torch.cat([xs, features_A, features_B], dim=1)
        theta = self.fusion(combined)
        theta = theta.view(-1, 2, 3)
        return theta

    def forward(self, img_A, img_B, src):
        identity_matrix = [1, 0, 0, 0, 1, 0]

        with autocast():
            # Extract features
            keypoints_A, descriptors_A = self.feature_extractor(img_A)
            keypoints_B, descriptors_B = self.feature_extractor(img_B)
            
            # Combine features
            features_A = descriptors_A.mean(dim=[2, 3])
            features_B = descriptors_B.mean(dim=[2, 3])
            
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

# Feature matching loss
def feature_matching_loss(keypoints_A, keypoints_B, theta):
    # Convert keypoints to coordinates
    batch_size = keypoints_A.size(0)
    loss = 0
    
    for b in range(batch_size):
        # Get top K keypoints
        kp_A = keypoints_A[b].view(-1)
        kp_B = keypoints_B[b].view(-1)
        
        # Get top K locations
        _, idx_A = torch.topk(kp_A, k=100)
        _, idx_B = torch.topk(kp_B, k=100)
        
        # Convert to 2D coordinates
        h, w = keypoints_A.size(2), keypoints_A.size(3)
        y_A = idx_A // w
        x_A = idx_A % w
        y_B = idx_B // w
        x_B = idx_B % w
        
        # Stack coordinates
        coords_A = torch.stack([x_A.float(), y_A.float()], dim=1)
        coords_B = torch.stack([x_B.float(), y_B.float()], dim=1)
        
        # Apply transformation to A coordinates
        theta_b = theta[b].view(2, 3)
        ones = torch.ones(coords_A.size(0), 1).cuda()
        coords_A_homo = torch.cat([coords_A, ones], dim=1)
        transformed_A = torch.mm(coords_A_homo, theta_b.t())
        
        # Calculate distance between transformed points and B points
        dist = torch.cdist(transformed_A, coords_B)
        min_dist, _ = torch.min(dist, dim=1)
        loss += min_dist.mean()
    
    return loss / batch_size

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