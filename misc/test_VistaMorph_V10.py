import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
import cv2
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import antialiased_cnns
from datasets_stn import * 
import kornia
import kornia.contrib as K

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to load")
parser.add_argument("--dataset_name", type=str, default="eurecom_warped_pairs", help="name of the dataset")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--experiment", type=str, default="tfcgan_stn_arar", help="experiment name")
opt = parser.parse_args()

os.makedirs("./images/test_results/%s" % opt.experiment, exist_ok=True)
cuda = True if torch.cuda.is_available() else False

# Set fixed random number seed
torch.manual_seed(42)

#################
# Edge Detection
################

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

#################
# FFT Components
################

class FFT_Components(object):
    def __init__(self, image):
        self.image = image

    def make_components(self):
        img = np.array(self.image)
        f_result = np.fft.rfft2(img)
        fshift = np.fft.fftshift(f_result)
        amp = np.abs(fshift)
        phase = np.arctan2(fshift.imag, fshift.real)
        return amp, phase

def fft_components(thermal_tensor):
    AMP = []
    PHA = []
    for t in range(0, thermal_tensor.size(0)):
        b = transforms.ToPILImage()(thermal_tensor[t, :, :, :]).convert("L")
        fft_space = FFT_Components(b)
        amp, phase = torch.Tensor(fft_space.make_components()).cuda()
        AMP.append(amp)
        PHA.append(phase)

    AMP_tensor = torch.cat(AMP).reshape(thermal_tensor.size(0), 1, opt.img_height, 129)
    PHA_tensor = torch.cat(PHA).reshape(thermal_tensor.size(0), 1, opt.img_height, 129)
    return AMP_tensor, PHA_tensor

#################
# ViT for STN
################

class LocalizerVIT(nn.Module):
    def __init__(self, img_shape):
        super(LocalizerVIT, self).__init__()
        channels, self.h, self.w = img_shape
        self.vit = nn.Sequential(
            K.VisionTransformer(image_size=self.h, patch_size=64, in_channels=channels*2)
        )

    def forward(self, x):
        with autocast():
            # Get ViT output [batch, 17, 768]
            out = self.vit(x).type(HalfTensor)
            # Reshape to match expected dimensions
            out = out.view(out.size(0), 1, 17, 768)
            print("out Vit:", out.size())
        return out

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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        input_shape = (opt.channels, opt.img_height, opt.img_width)
        self.localization = LocalizerVIT(input_shape)
        
        # Simplified fusion layer without feature information
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
            img_input = torch.cat((img_A, img_B), 1)
            dtheta = self.stn_phi(img_input)
            
            identity_theta = torch.tensor(identity_matrix, dtype=torch.float).cuda()
            dtheta = dtheta.reshape(img_A.size(0), 2*3)
            theta = dtheta + identity_theta.unsqueeze(0).repeat(img_A.size(0),1)

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
# Generator
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
# Configure data loader
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

#=================================
# Load models
#=================================

# Initialize models
input_shape = (opt.channels, opt.img_height, opt.img_width)
model = Net()
generator2 = GeneratorUNet2(input_shape)

if cuda:
    model = model.cuda()
    generator2 = generator2.cuda()

# Load trained weights
model.load_state_dict(torch.load("./saved_models/%s/net_%d.pth" % (opt.experiment, opt.epoch)))
generator2.load_state_dict(torch.load("./saved_models/%s/generator2_%d.pth" % (opt.experiment, opt.epoch)))

# Set to eval mode
model.eval()
generator2.eval()

# Tensor type
HalfTensor = torch.cuda.HalfTensor if cuda else torch.HalfTensor

#=================================
# Test
#=================================

def denormalize(x):
    return (x + 1) / 2.0

def calculate_affine_error(pred_theta, gt_theta):
    """Calculate the error between predicted and ground truth affine matrices.
    
    Args:
        pred_theta (numpy.ndarray): Predicted affine matrix [2x3]
        gt_theta (numpy.ndarray): Ground truth affine matrix [2x3]
        
    Returns:
        float: Mean error across all parameters
    """
    # Convert to full 3x3 matrices
    pred_full = np.eye(3)
    pred_full[:2, :] = pred_theta
    
    gt_full = np.eye(3)
    gt_full[:2, :] = gt_theta
    
    # Calculate inverse of ground truth
    gt_inv = np.linalg.inv(gt_full)
    
    # Calculate error matrix
    error_matrix = np.matmul(pred_full, gt_inv)
    
    # Extract translation error
    tx_error = error_matrix[0, 2]
    ty_error = error_matrix[1, 2]
    
    # Extract rotation error (in degrees)
    rotation_error = np.abs(np.arctan2(error_matrix[1, 0], error_matrix[0, 0]) * 180 / np.pi)
    
    # Extract scale error
    scale_x = np.sqrt(error_matrix[0, 0]**2 + error_matrix[1, 0]**2)
    scale_y = np.sqrt(error_matrix[0, 1]**2 + error_matrix[1, 1]**2)
    scale_error = np.abs(scale_x - 1) + np.abs(scale_y - 1)
    
    # Calculate final score
    score = (np.abs(tx_error) + np.abs(ty_error) + rotation_error + scale_error) / 4
    
    return score

def test():
    # Create directory for predictions
    os.makedirs("./images/test_results/%s/preds" % opt.experiment, exist_ok=True)
    
    # Initialize scoring metrics
    total_score = 0
    num_samples = 0
    
    for i, batch in enumerate(test_dataloader):
        real_A = Variable(batch["A"].type(HalfTensor))
        real_B = Variable(batch["B"].type(HalfTensor))
        gt_theta = batch["Y"].numpy()  # Ground truth affine matrix
        
        # Get edge maps
        edge_A = edge_detection(real_A)
        edge_B = edge_detection(real_B)
        
        # Spatial transformation
        warped_B, theta = model(img_A=edge_A, img_B=edge_B, src=real_B)
        
        # Calculate error score
        pred_theta_np = theta.cpu().numpy()[0]  # Get first batch item
        gt_theta_np = gt_theta[0]  # Get first batch item
        score = calculate_affine_error(pred_theta_np, gt_theta_np)
        total_score += score
        num_samples += 1
        
        # Save the predicted affine matrix
        formatted_theta = f"{pred_theta_np[0]:.6f} {pred_theta_np[1]:.6f} {pred_theta_np[2]:.6f} {pred_theta_np[3]:.6f} {pred_theta_np[4]:.6f} {pred_theta_np[5]:.6f} 0 0 1"
        with open(f"./images/test_results/{opt.experiment}/preds/predicted_theta_{i:05d}.txt", "w") as f:
            f.write(formatted_theta)
        
        # Generate fake_A1
        fake_A1 = generator2(warped_B)
        edge_fake_A1 = edge_detection(fake_A1)
        
        # Get FFT components
        Amp_w, Pha_w = fft_components(warped_B)
        Amp_r, Pha_r = fft_components(real_A)
        
        # Denormalize images
        real_A_denorm = denormalize(real_A.data)
        real_B_denorm = denormalize(real_B.data)
        warped_B_denorm = denormalize(warped_B.data)
        fake_A1_denorm = denormalize(fake_A1.data)
        edge_A_denorm = denormalize(edge_A.data)
        edge_fake_A1_denorm = denormalize(edge_fake_A1.data)
        
        # Stack images
        img_sample = torch.cat((
            real_A_denorm, real_B_denorm, warped_B_denorm,
            fake_A1_denorm, edge_A_denorm, edge_fake_A1_denorm
        ), -1)
        
        # Save images
        save_image(img_sample, "./images/test_results/%s/%d.png" % (opt.experiment, i), nrow=3, normalize=False)
        
        # Print progress with current score
        sys.stdout.write("\r[Test] [%d/%d] [Score: %.4f]" % (i, len(test_dataloader), score))
        sys.stdout.flush()
    
    # Calculate and print final average score
    final_score = total_score / num_samples
    print(f"\nFinal Average Score: {final_score:.4f}")
    
    # Save final score to file
    with open(f"./images/test_results/{opt.experiment}/final_score.txt", "w") as f:
        f.write(f"Final Average Score: {final_score:.4f}\n")
        f.write(f"Number of samples: {num_samples}\n")

if __name__ == "__main__":
    test()

