z"""
VistaMorph V8: Enhanced Registration with Cycle Consistency and Identity Preservation

This version implements several key improvements to handle extreme warps between real_A and real_B:
1. Cycle consistency loss to ensure reversible transformations
2. Identity preservation loss to maintain real_A features in fake_A1
3. Modified loss weights to balance registration accuracy with feature preservation

Key changes from V6:
- Added cycle consistency to ensure transformations are reversible
- Added identity loss to maintain real_A features in fake_A1
- Adjusted loss weights to better balance registration and feature preservation
- Enhanced comments explaining the registration process

The goal is to improve the accuracy of the affine matrix (theta) prediction
while maintaining the quality of fake_A generation, which is crucial for
precise registration between real_A and real_B.
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

# [Previous code remains the same until the training loop]

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
            warped_B, theta = model(img_A=real_A, img_B=real_A, src=real_B)
            
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

            # Morphological triplet loss to reinforce feature preservation
            morph_loss = morph_triplet(real_A, real_B, fake_A1)

            # Adversarial losses
            loss_GAN1 = global_gen_loss(real_A, warped_B, fake_B, mode='B')
            loss_GAN2 = global_gen_loss(real_A, warped_B, fake_A1, mode='A')
            loss_GAN = (loss_GAN1 + loss_GAN2).mean()

            # Tie point loss - MSE between predicted and ground truth affine matrices
            tie_loss = criterion_MSE(theta, Y)

            # Total Loss with adjusted weights
            alpha1 = 0.05  # Cycle consistency weight
            alpha2 = 0.25  # Tie point loss weight
            alpha3 = 0.1   # Identity loss weight
            loss_G = (loss_GAN + recon_loss + perc_loss + morph_loss + 
                     alpha1*cycle_loss + alpha2*tie_loss + alpha3*identity_loss).mean()

        scaler.scale(loss_G).backward()
        scaler.step(optimizer_G)
        print("+ + + optimizer_G.step() + + + ")

        # [Rest of the training loop remains the same] 