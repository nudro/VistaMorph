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
from datasets_stn_with_labels_A1 import * # only A and B
import kornia
import kornia.feature as KF
from kornia.feature import match_adalam  # Changed from AdaLAM to match_adalam
from lpips import LPIPS
from kornia import morphology as morph
import kornia.contrib as K
import matplotlib.pyplot as plt

def adalam(warped_B, real_A, device, return_matches=False):
    """Perform keypoint detection and matching using KeyNet and AdaLAM.
    Args:
        warped_B: Warped image tensor [B, C, H, W]
        real_A: Real image tensor [B, C, H, W]
        device: Device to use for tensor operations
        return_matches: Whether to return matches and keypoints for visualization
    Returns:
        If return_matches is False:
            torch.Tensor: Homography loss
        If return_matches is True:
            tuple: (matches, kpts_warped, kpts_real)
    """
    batch_size = warped_B.size(0)
    keynet = kornia.feature.KeyNetDetector()
    
    # Initialize lists to store results for each image in batch
    all_matches = []
    all_kpts_warped = []
    all_kpts_real = []
    total_homog_loss = torch.tensor(0.0, device=device)
    
    # Process each image in the batch
    for i in range(batch_size):
        # Get single images from batch
        warped_B_single = warped_B[i:i+1]  # Keep batch dimension
        real_A_single = real_A[i:i+1]      # Keep batch dimension
        
        # Convert to grayscale and float32
        warped_B_gray = kornia.color.rgb_to_grayscale(warped_B_single).float()
        real_A_gray = kornia.color.rgb_to_grayscale(real_A_single).float()
        
        # Detect keypoints using KeyNet
        kpts_warped, desc_warped = keynet(warped_B_gray)
        kpts_real, desc_real = keynet(real_A_gray)
        
        # Match keypoints using AdaLAM (changed from class to function)
        matches = match_adalam(desc_warped, desc_real, kpts_warped, kpts_real)
        
        if return_matches:
            all_matches.append(matches)
            all_kpts_warped.append(kpts_warped)
            all_kpts_real.append(kpts_real)
        else:
            # Calculate homography loss for this image
            homog_loss = homog_loss(matches, kpts_warped, kpts_real, device)
            total_homog_loss += homog_loss
    
    if return_matches:
        # For visualization, return results from first image in batch
        return all_matches[0], all_kpts_warped[0], all_kpts_real[0]
    else:
        # Return average homography loss across batch
        return total_homog_loss / batch_size 