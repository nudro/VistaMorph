"""
Test script for VistaMorph V10: Registration with Cycle Consistency

This script tests the V10 model which includes:
1. Edge-based registration
2. FFT components for frequency domain alignment
3. Simplified generator and discriminator setup
4. Geometric tie-loss for better transformation accuracy
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
import kornia.contrib as K
from datasets_stn_with_labels import *

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=200, help="epoch to load checkpoint from")
parser.add_argument("--dataset_name", type=str, default="eurecom_warped_pairs", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--gpu_num", type=int, default=0, help="gpu card")
parser.add_argument("--experiment", type=str, default="tfcgan_stn_arar", help="experiment name")
opt = parser.parse_args()

cuda = True if torch.cuda.is_available() else False
torch.cuda.set_device(opt.gpu_num)
torch.manual_seed(42)

######################
# Edge Detection
######################

def edge_detection(img):
    """Extract edges from image using combined Canny and Laplacian operators.
    Args:
        img (torch.Tensor): Input image tensor [B, C, H, W]
    Returns:
        torch.Tensor: Combined edge map [B, 1, H, W]
    """
    # Convert to grayscale
    gray = kornia.color.rgb_to_grayscale(img)
    
    # Canny edge detection
    canny = kornia.filters.canny(gray, low_threshold=0.1, high_threshold=0.3)[0]
    
    # Laplacian edge detection
    laplacian = kornia.filters.laplacian(gray, kernel_size=3)
    
    # Combine and normalize
    return torch.clamp(canny + laplacian, 0, 1)

######################
# Model Architecture
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
        # Use 6 channels for expanded edge images
        self.vit = nn.Sequential(
            K.VisionTransformer(image_size=self.h, patch_size=64, in_channels=6)
        )

    def forward(self, x):
        with autocast():
            # Get ViT output [batch, 17, 768]
            out = self.vit(x).type(HalfTensor)
            # Reshape to match expected dimensions
            out = out.view(out.size(0), 1, 17, 768)
        return out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        input_shape = (opt.channels, opt.img_height, opt.img_width)
        self.localization = LocalizerVIT(input_shape)
        
        # Simplified fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(1*17*768, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 256),
            nn.ReLU(True),
            ResidualBlock(256),
            nn.Linear(256, 3*2),
            nn.Sigmoid()
        )
        
    def stn_phi(self, x):
        xs = self.localization(x)  # [batch, 1, 17, 768]
        xs = xs.view(-1, 1 * xs.size(2) * xs.size(3))  # [batch, 17*768]
        theta = self.fusion(xs)
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
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=1,
)

# Initialize model
model = Net()
if cuda:
    model = model.cuda()
    model = torch.nn.DataParallel(model, device_ids=[0,1])

# Load pretrained model
model.load_state_dict(torch.load("./saved_models/%s/net_%d.pth" % (opt.experiment, opt.epoch)))

# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
HalfTensor = torch.cuda.HalfTensor if cuda else torch.HalfTensor

##############################
# Testing
##############################

def calculate_affine_error(pred_theta, target_theta):
    """Calculate error between predicted and ground truth affine matrices.
    Args:
        pred_theta: Predicted transformation matrix [B, 2, 3]
        target_theta: Target transformation matrix [B, 2, 3]
    Returns:
        float: Average error score
    """
    # Convert to full 3x3 matrices
    pred_full = torch.eye(3).unsqueeze(0).repeat(pred_theta.size(0), 1, 1).cuda()
    pred_full[:, :2, :] = pred_theta
    
    target_full = torch.eye(3).unsqueeze(0).repeat(target_theta.size(0), 1, 1).cuda()
    target_full[:, :2, :] = target_theta
    
    # Calculate inverse of ground truth
    target_inv = torch.inverse(target_full)
    
    # Calculate error matrix
    error_matrix = torch.bmm(pred_full, target_inv)
    
    # Extract translation, rotation, and scale errors
    trans_error = torch.norm(error_matrix[:, :2, 2], dim=1)
    rot_error = torch.abs(torch.atan2(error_matrix[:, 1, 0], error_matrix[:, 0, 0]))
    scale_error = torch.abs(torch.norm(error_matrix[:, :2, 0], dim=1) - 1)
    
    # Calculate average score
    score = (trans_error + rot_error + scale_error).mean()
    
    return score.item()

def test():
    # Create directory for predictions
    os.makedirs("./images/test_results/%s/preds" % opt.experiment, exist_ok=True)
    
    # Initialize scoring metrics
    total_score = 0
    num_samples = 0
    
    for i, batch in enumerate(test_dataloader):
        real_A = Variable(batch["A"].type(HalfTensor))
        real_B = Variable(batch["B"].type(HalfTensor))
        Y = Variable(batch["Y"].type(HalfTensor))  # Ground truth affine matrix
        
        # Get edge-detected images
        edge_A = edge_detection(real_A)
        edge_B = edge_detection(real_B)
        
        # Get predictions
        warped_B, theta = model(img_A=edge_A, img_B=edge_B, src=real_B)
        
        # Calculate error score
        score = calculate_affine_error(theta.view(-1, 2, 3), Y.view(-1, 2, 3))
        total_score += score
        num_samples += 1
        
        # Save predicted affine matrix
        pred_theta = theta[0].cpu().numpy()
        np.savetxt(
            "./images/test_results/%s/preds/predicted_theta_%05d.txt" % (opt.experiment, i),
            pred_theta,
            fmt='%.6f',
            header='a b tx\nc d ty\n0 0 1'
        )
        
        # Save images
        def denormalize(x):
            return (x + 1) / 2.0
        
        real_A_denorm = denormalize(real_A.data)
        real_B_denorm = denormalize(real_B.data)
        warped_B_denorm = denormalize(warped_B.data)
        edge_A_denorm = denormalize(edge_A.data)
        edge_B_denorm = denormalize(edge_B.data)
        
        # Stack images
        img_sample = torch.cat((
            real_A_denorm, real_B_denorm, warped_B_denorm,
            edge_A_denorm, edge_B_denorm
        ), -1)
        
        # Save with proper normalization
        save_image(img_sample, "./images/test_results/%s/%d.png" % (opt.experiment, i), nrow=1, normalize=False)
        
        # Print progress
        print("[Batch %d/%d] [Score: %f]" % (i, len(test_dataloader), score))
    
    # Calculate and save final score
    final_score = total_score / num_samples
    with open("./images/test_results/%s/final_score.txt" % opt.experiment, "w") as f:
        f.write("Final Score: %f\n" % final_score)
    print("Final Score: %f" % final_score)

if __name__ == "__main__":
    test()

