"""
Test script for VistaMorph V10: Registration with Cycle Consistency

This script tests the V10 model which includes:
1. Edge-based registration
2. FFT components for frequency domain alignment
3. Simplified generator and discriminator setup
4. Geometric tie-loss for better transformation accuracy
5. Homography loss using KeyNet and AdaLAM
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.cuda.amp import autocast
import kornia
import matplotlib.pyplot as plt
from datasets_stn_with_labels import *
from vistamorph_A1 import Net, edge_detection, affine_to_tiepoints, adalam, homog_loss

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="eurecom_warped_pairs", help="name of the dataset")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--gpu_num", type=int, default=0, help="gpu card")
parser.add_argument("--experiment", type=str, default="tfcgan_stn_arar", help="experiment name")
parser.add_argument("--epoch", type=int, default=200, help="epoch to load model from")
opt = parser.parse_args()

os.makedirs("./test_images/%s" % opt.experiment, exist_ok=True)
os.makedirs("./test_images/%s/matches" % opt.experiment, exist_ok=True)

cuda = True if torch.cuda.is_available() else False
torch.cuda.set_device(opt.gpu_num)

##############################
# Transforms and Dataloaders
##############################

transforms_ = [
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

test_dataloader = DataLoader(
    TestImageDataset(root = "./data/%s" % opt.dataset_name,
        transforms_=transforms_,
        mode="test"),
    batch_size=1,
    shuffle=True,
    num_workers=1,
)

# ===========================================================
# Initialize model
# ===========================================================
input_shape_patch = (opt.channels, opt.img_height, opt.img_width)
model = Net()

if cuda:
    model = model.cuda()

# Load trained model
model.load_state_dict(torch.load("./saved_models/%s/net_%d.pth" % (opt.experiment, opt.epoch)))
model.eval()

# Tensor type
HalfTensor = torch.cuda.HalfTensor if cuda else torch.HalfTensor

def denormalize(x):
    """Convert normalized tensor to image format"""
    return (x + 1) / 2.0

def visualize_matches(real_A, real_B, warped_B, matches, kpts_warped, kpts_real, i):
    """Visualize keypoint matches between images.
    Args:
        real_A: Real A image tensor
        real_B: Real B image tensor
        warped_B: Warped B image tensor
        matches: Keypoint matches
        kpts_warped: Keypoints from warped image
        kpts_real: Keypoints from real image
        i: Sample index
    """
    # Convert tensors to numpy for visualization
    real_A_np = real_A[0].cpu().numpy().transpose(1, 2, 0)
    real_B_np = real_B[0].cpu().numpy().transpose(1, 2, 0)
    warped_B_np = warped_B[0].cpu().numpy().transpose(1, 2, 0)
    
    # Create figure for matches visualization
    plt.figure(figsize=(18, 6))
    plt.subplot(131)
    plt.imshow(real_A_np)
    plt.title('Real A')
    plt.subplot(132)
    plt.imshow(real_B_np)
    plt.title('Real B')
    plt.subplot(133)
    plt.imshow(warped_B_np)
    plt.title('Warped B')
    
    # Plot matches
    for match in matches[:20]:  # Plot first 20 matches
        src_pt = kpts_warped[match[0]].cpu().numpy()
        dst_pt = kpts_real[match[1]].cpu().numpy()
        plt.plot([src_pt[0], dst_pt[0]], [src_pt[1], dst_pt[1]], 'r-', alpha=0.5)
    
    plt.savefig(f"test_images/{opt.experiment}/matches/{i}.png")
    plt.close()

##############################
#       Testing
##############################

for i, batch in enumerate(test_dataloader):
    real_A = Variable(batch["A"].type(HalfTensor))
    real_B = Variable(batch["B"].type(HalfTensor))
    Y = Variable(batch["Y"].type(HalfTensor))  # Ground truth affine matrix

    with torch.no_grad():
        with autocast():
            # Get edge-detected images
            edge_A = edge_detection(real_A)
            edge_B = edge_detection(real_B)
            
            # Spatial transformation to align real_B with real_A
            warped_B, theta = model(img_A=edge_A, img_B=edge_B, src=real_B)
            
            # Convert theta to tiepoints for visualization
            pred_source, pred_target = affine_to_tiepoints(theta.view(-1, 2, 3))
            gt_source, gt_target = affine_to_tiepoints(Y.view(-1, 2, 3))
            
            # Calculate tiepoint error
            tie_error = F.smooth_l1_loss(pred_target, gt_target).item()
            
            # Calculate homography loss and get matches for visualization
            matches, kpts_warped, kpts_real = adalam(warped_B, real_A, warped_B.device, return_matches=True)
            homog_error = homog_loss(matches, kpts_warped, kpts_real, warped_B.device).item()
            
            print(f"Sample {i+1} - Tiepoint Error: {tie_error:.4f}, Homography Error: {homog_error:.4f}")

    # Denormalize images for visualization
    real_A_denorm = denormalize(real_A.data)
    real_B_denorm = denormalize(real_B.data)
    warped_B_denorm = denormalize(warped_B.data)

    # Save images
    img_sample = torch.cat((
        real_A_denorm, real_B_denorm, warped_B_denorm
    ), -1)
    
    save_image(img_sample, "./test_images/%s/%d.png" % (opt.experiment, i), nrow=3, normalize=False)
    
    # Visualize keypoint matches
    if matches is not None and len(matches) > 0:
        visualize_matches(real_A_denorm, real_B_denorm, warped_B_denorm, matches, kpts_warped, kpts_real, i)

