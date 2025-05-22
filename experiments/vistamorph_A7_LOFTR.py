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
from kornia_moons.viz import draw_LAF_matches


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
criterion_GAN = torch.nn.BCEWithLogitsLoss() # Relativitic

criterion_lpips = LPIPS(
    net='vgg',  # choose a network type from ['alex', 'squeeze', 'vgg']
    version='0.1'  # Currently, v0.1 is supported
)

criterion_L1 = nn.L1Loss()
criterion_amp = nn.L1Loss()   # For FFT amplitude loss
criterion_phase = nn.L1Loss() # For FFT phase loss


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
  
    # Cast real_A and real_B into a set of canny edges or laplacaian edges
    edge_A, edge_B = edge_detection(real_A), edge_detection(real_B)
    print("edge_A:", edge_A.shape)
    edge_A = edge_A.expand(-1, 3, -1, -1)
    print("edge_A:", edge_A.shape)
            
    fake_A = generator2(real_B)
    fake_B = generator1(edge_A)
    #warped_B, theta = model(img_A=fake_A, img_B=real_A, src=real_B)  
    warped_B, theta = model(img_A=real_A, img_B=edge_A, src=real_B)  

    """

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
    """
    
    # Save the regular image comparison
    img_sample_global = torch.cat((warped_B.data, real_A.data, real_B.data, fake_A.data, fake_B.data), -1)
    save_image(img_sample_global, "./images/%s/%s.png" % (opt.experiment, batches_done), nrow=4, normalize=True)

    """
    # Create and save match visualization using Kornia's tools
    mkpts0 = original_features['keypoints0'].cuda()  # Ensure on GPU
    mkpts1 = original_features['keypoints1'].cuda()  # Ensure on GPU
    
    # Check if we have any matches
    if len(mkpts0) > 0:
        print(f"\nNumber of matches found: {len(mkpts0)}")
        print(f"mkpts0 shape: {mkpts0.shape}, mkpts1 shape: {mkpts1.shape}")
        
        # Convert to LAF format for visualization
        laf0 = KF.laf_from_center_scale_ori(
            mkpts0.view(1, -1, 2),
            torch.ones(mkpts0.shape[0], device='cuda').view(1, -1, 1, 1),  # Ensure on GPU
            torch.ones(mkpts0.shape[0], device='cuda').view(1, -1, 1)      # Ensure on GPU
        )
        
        laf1 = KF.laf_from_center_scale_ori(
            mkpts1.view(1, -1, 2),
            torch.ones(mkpts1.shape[0], device='cuda').view(1, -1, 1, 1),  # Ensure on GPU
            torch.ones(mkpts1.shape[0], device='cuda').view(1, -1, 1)      # Ensure on GPU
        )
        
        print(f"LAF shapes - laf0: {laf0.shape}, laf1: {laf1.shape}")
        
        # Create indices for matches
        idx = torch.arange(mkpts0.shape[0], device='cuda').view(-1, 1).repeat(1, 2)  # Ensure on GPU
        print(f"Match indices shape: {idx.shape}")
        
        # Move all tensors to CPU before visualization
        laf0 = laf0.cpu()
        laf1 = laf1.cpu()
        idx = idx.cpu()
        
        # Normalize images from [-1,1] to [0,1] range and convert to numpy
        print(f"\nOriginal tensor shapes - real_A: {real_A.shape}, real_B: {real_B.shape}")
        img1 = ((real_A.cpu().squeeze(0) + 1) / 2).permute(1, 2, 0).numpy().astype(np.float32)  # Ensure float32
        img2 = ((real_B.cpu().squeeze(0) + 1) / 2).permute(1, 2, 0).numpy().astype(np.float32)  # Ensure float32
        print(f"Converted numpy array shapes - img1: {img1.shape}, img2: {img2.shape}")
        print(f"Image value ranges - img1: [{img1.min():.3f}, {img1.max():.3f}], img2: [{img2.min():.3f}, {img2.max():.3f}]")
        print(f"Image dtypes - img1: {img1.dtype}, img2: {img2.dtype}")
        
        # Draw matches
        plt.figure(figsize=(12, 6))
        print("\nDrawing matches with draw_LAF_matches...")
        draw_LAF_matches(
            laf0,
            laf1,
            idx,
            img1,  # Use numpy array directly
            img2,  # Use numpy array directly
            torch.ones(mkpts0.shape[0], dtype=torch.bool, device='cuda').cpu(),  # Move to CPU before passing to draw_LAF_matches
            draw_dict={
                "inlier_color": (0.2, 1, 0.2),
                "tentative_color": (1.0, 0.5, 1),
                "feature_color": (0.2, 0.5, 1),
                "vertical": False
            }
        )
    else:
        print("\nNo matches found, displaying side-by-side images")
        # If no matches, just show the images side by side
        plt.figure(figsize=(12, 6))
        
        # Normalize images from [-1,1] to [0,1] range and convert to numpy
        print(f"\nOriginal tensor shapes - real_A: {real_A.shape}, real_B: {real_B.shape}")
        img1 = ((real_A.cpu().squeeze(0) + 1) / 2).permute(1, 2, 0).numpy().astype(np.float32)  # Ensure float32
        img2 = ((real_B.cpu().squeeze(0) + 1) / 2).permute(1, 2, 0).numpy().astype(np.float32)  # Ensure float32
        print(f"Converted numpy array shapes - img1: {img1.shape}, img2: {img2.shape}")
        print(f"Image value ranges - img1: [{img1.min():.3f}, {img1.max():.3f}], img2: [{img2.min():.3f}, {img2.max():.3f}]")
        print(f"Image dtypes - img1: {img1.dtype}, img2: {img2.dtype}")
        
        plt.subplot(121)
        plt.imshow(img1)
        plt.title('Source Image')
        plt.axis('off')
        
        plt.subplot(122)
        plt.imshow(img2)
        plt.title('Target Image')
        plt.axis('off')
    
    print(f"\nSaving visualization to ./images/{opt.experiment}/matches_{batches_done}.png")
    plt.savefig(f"./images/{opt.experiment}/matches_{batches_done}.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Visualization complete\n")
    """
class FFT_Components(object):
    def __init__(self, image):
        self.image = image

    def make_components(self):
        img = np.array(self.image) #turn into numpy
        f_result = np.fft.rfft2(img)
        fshift = np.fft.fftshift(f_result)
        amp = np.abs(fshift)
        phase = np.arctan2(fshift.imag,fshift.real)
        return amp, phase

    def make_spectra(self):
        img = np.array(self.image) #turn into numpy
        f_result = np.fft.fft2(img) # setting this to regular FFT2 to make magnitude spectra
        fshift = np.fft.fftshift(f_result)
        magnitude_spectrum = np.log(np.abs(fshift))
        return magnitude_spectrum

def fft_components(thermal_tensor):
    # thermal_tensor can be fake_B or real_B
    AMP = []
    PHA = []
    for t in range(0, thermal_tensor.size(0)):
        # Convert to grayscale (1 channel)
        b = transforms.ToPILImage()(thermal_tensor[t, :, :, :]).convert("L")
        fft_space = FFT_Components(b)
        amp, phase = torch.Tensor(fft_space.make_components()).cuda() # convert them into torch tensors
        AMP.append(amp)
        PHA.append(phase)

    # reshape each amplitude and phase to Torch tensors
    # For FFT2 the dims is 256 x 129 for half of all real values + 1 col due to Hermitian Symmetry
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
            K.VisionTransformer(image_size=self.h, patch_size=16, in_channels=channels*2) # try 16
        )

    def forward(self, x):
        # if you try different patch sizes, adjust tensors
        # patch 16: torch.Size([batch, 257, 768])
        # patch 32: torch.Size([batch, 65, 768])
        # patch 64: torch.Size([batch, 17, 768])
        with autocast():
            out = self.vit(x).type(HalfTensor) # returns at patch_size = 16, torch.Size([batch, 257, 768])
            #print("out Vit:", out.size())
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

        """
        # original
        self.fc_loc = nn.Sequential(
            nn.Linear(1*17*768, 1024),  #original, patch=64, nn.Linear(1*17*768, 1024)
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 3*2),
            nn.Sigmoid())
        """
        self.fc_loc = nn.Sequential(
            nn.Linear(1*257*768, 4096),  #original, patch=64, nn.Linear(1*17*768, 1024)
            nn.ReLU(True),
            nn.Linear(4096, 2048),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 256),
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

                # add regularization? ?
                grid_displacement = torch.norm(rs_grid - rs_grid.mean(dim=[1,2], keepdim=True), dim=-1)
                
                # Adaptive noise scaling based on displacement
                noise_scale = torch.clamp(0.01 * grid_displacement, max=0.05)
                noise = torch.randn_like(rs_grid) * noise_scale.unsqueeze(-1)
                rs_grid = rs_grid + noise
                
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


"""
loss_GAN1 = global_gen_loss(real_A, warped_B, fake_B, mode='B') # (A,WB), (A, FB)
loss_GAN2 = global_gen_loss(real_A, warped_B, fake_A, mode='A')
"""

def global_disc_loss(real_A, real_B, fake_img, mode):
    # GAN 1 == B
    if mode=='B':
        pred_real = discriminator1(real_B, real_A)
        pred_fake = discriminator1(fake_img.detach(), real_A)
        loss_real = criterion_GAN(pred_real - pred_fake, valid)
        loss_fake = criterion_GAN(pred_fake - pred_real, fake)
        loss_D = (loss_real + loss_fake).mean()

    # GAN 2 == A
    if mode=='A':
        pred_real = discriminator2(real_A, real_B)
        pred_fake = discriminator2(fake_img.detach(), real_B)
        loss_real = criterion_GAN(pred_real - pred_fake, valid)
        loss_fake = criterion_GAN(pred_fake - pred_real, fake)
        loss_D = (loss_real + loss_fake).mean()

    return loss_D

def get_perceptual_weight(epoch, max_epochs=210, warmup_epochs=10):
    """
    Calculate the weight for perceptual loss with warmup and sigmoid-based softening.
    Args:
        epoch: Current epoch
        max_epochs: Total number of epochs
        warmup_epochs: Number of epochs for warmup period
    Returns:
        weight: Float between 0 and 0.8
    """
    if epoch < warmup_epochs:
        # Linear increase from 0 to 1 during warmup
        return epoch / warmup_epochs
    else:
        # Sigmoid function that smoothly approaches 0.8
        x = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
        return 0.8 * (1 / (1 + math.exp(-5 * (x - 0.5))))

def find_common_matches(points1, points2, threshold=20.0):
    """
    Find common matches between two sets of keypoints based on Euclidean distance.
    Args:
        points1: First set of keypoints [N, 2]
        points2: Second set of keypoints [M, 2]
        threshold: Maximum distance to consider points as matching
    Returns:
        indices1: Indices of matching points in points1
        indices2: Indices of matching points in points2
    """
    # Calculate pairwise distances
    dist_matrix = torch.cdist(points1, points2)
    
    # Find minimum distance for each point in points1
    min_dist1, idx2 = torch.min(dist_matrix, dim=1)
    
    # Find minimum distance for each point in points2
    min_dist2, idx1 = torch.min(dist_matrix, dim=0)
    
    # Find points that are mutual nearest neighbors
    mutual_matches = []
    for i in range(len(points1)):
        j = idx2[i]
        if idx1[j] == i and min_dist1[i] < threshold and min_dist2[j] < threshold:
            mutual_matches.append((i, j))
    
    if not mutual_matches:
        return None, None
    
    indices1, indices2 = zip(*mutual_matches)
    return torch.tensor(indices1, device=points1.device), torch.tensor(indices2, device=points2.device)

def loftr_feature_loss(img1, img2, epoch, max_epochs=210, min_matches=5, confidence_threshold=0.2):
    """
    Calculate LoFTR feature loss between two images with warm-up and fallback mechanisms.
    Args:
        img1: First image tensor [1, 3, H, W]
        img2: Second image tensor [1, 3, H, W]
        epoch: Current training epoch
        max_epochs: Total number of epochs for warm-up
        min_matches: Minimum number of matches required
        confidence_threshold: Minimum confidence score for matches
    Returns:
        loss: Combined LoFTR feature loss or fallback loss
    """
    # Warm-up weight calculation (linear increase from 0 to 1 over first 20% of epochs)
    warmup_epochs = int(max_epochs * 0.2)
    if epoch < warmup_epochs:
        weight = epoch / warmup_epochs
    else:
        weight = 1.0
    
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
    
    # Apply confidence threshold
    forward_mask = conf_forward > confidence_threshold
    backward_mask = conf_backward > confidence_threshold
    
    # Check if we have enough matches
    if forward_mask.sum() < min_matches or backward_mask.sum() < min_matches:
        # Fallback to L1 loss if not enough matches
        return criterion_L1(img1, img2)
    
    # Filter matches by confidence
    mkpts0_forward = mkpts0_forward[forward_mask]
    mkpts1_forward = mkpts1_forward[forward_mask]
    conf_forward = conf_forward[forward_mask]
    
    mkpts0_backward = mkpts0_backward[backward_mask]
    mkpts1_backward = mkpts1_backward[backward_mask]
    conf_backward = conf_backward[backward_mask]

    # Find common matches between forward and backward directions
    indices_forward, indices_backward = find_common_matches(mkpts0_forward, mkpts1_forward)
    
    if indices_forward is None or len(indices_forward) < min_matches:
        # Fallback to L1 loss if not enough common matches
        return criterion_L1(img1, img2)
    
    # Use only common matches
    mkpts0_forward = mkpts0_forward[indices_forward]
    mkpts1_forward = mkpts1_forward[indices_forward]
    conf_forward = conf_forward[indices_forward]
    
    mkpts0_backward = mkpts0_backward[indices_backward]
    mkpts1_backward = mkpts1_backward[indices_backward]
    conf_backward = conf_backward[indices_backward]

    # 1. Mean Euclidean Distance (MED)
    dist_forward = torch.norm(mkpts0_forward - mkpts1_forward, dim=1)
    dist_backward = torch.norm(mkpts0_backward - mkpts1_backward, dim=1)
    med_loss = (dist_forward.mean() + dist_backward.mean()) / 2
    
    # 2. Symmetric Transfer Error
    sym_error = torch.abs(dist_forward - dist_backward).mean()
    
    # 3. Confidence-weighted Error
    conf_weighted_forward = (conf_forward * dist_forward).sum() / conf_forward.sum()
    conf_weighted_backward = (conf_backward * dist_backward).sum() / conf_backward.sum()
    conf_weighted_loss = (conf_weighted_forward + conf_weighted_backward) / 2
    
    # Combine losses with weights and warm-up
    total_loss = (med_loss + 0.5 * sym_error + 0.5 * conf_weighted_loss) * weight
    
    return total_loss

# ===========================================================
# Initialize generator and discriminator
# ===========================================================

# Gen1/Disc1 == B, 
# Gen2/Disc2 = A
input_shape_patch = (opt.channels, opt.img_height, opt.img_width)
generator1 = GeneratorUNet1(input_shape_patch) # for fake_A
discriminator1 = Discriminator1(input_shape_patch) # for fake_A

generator2 = GeneratorUNet2(input_shape_patch) # for fake_A
discriminator2 = Discriminator2(input_shape_patch) # for fake_A

if cuda:
    generator1 = generator1.cuda()
    discriminator1 = discriminator1.cuda()
    generator2 = generator2.cuda()
    discriminator2 = discriminator2.cuda()
    model = Net().cuda()

    criterion_GAN.cuda()
    criterion_L1.cuda()
    criterion_lpips.cuda()
    criterion_amp.cuda()
    criterion_phase.cuda()

# Trained on multigpus - change your device ids if needed
generator1 = torch.nn.DataParallel(generator1, device_ids=[0, 1])
generator2 = torch.nn.DataParallel(generator2, device_ids=[0, 1])
discriminator1 = torch.nn.DataParallel(discriminator1, device_ids=[0, 1])
discriminator2 = torch.nn.DataParallel(discriminator2, device_ids=[0, 1])
model = torch.nn.DataParallel(model, device_ids=[0, 1])

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

optimizer_G = torch.optim.Adam(itertools.chain(generator1.parameters(), generator2.parameters(),  model.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_M = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(itertools.chain(discriminator1.parameters(), discriminator2.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2))

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

def affine_to_tiepoints(theta, img_size=(256, 256)):
    """
    Convert affine transformation matrix to tiepoints.
    Args:
        theta: Affine transformation matrix [B, 2, 3] or [2, 3]
        img_size: Tuple of (height, width) of the image
    Returns:
        source_points: Source points in the original image [B, 4, 2]
        target_points: Target points after transformation [B, 4, 2]

    pred_source, pred_target = affine_to_tiepoints(theta.view(-1, 2, 3))
    gt_source, gt_target = affine_to_tiepoints(Y.view(-1, 2, 3))
    """
    #print("--------AFFINE TO TIEPOINTS---------")
    #print(theta)
    #print(".......")
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

def edge_detection(img):
    """Extract edges from image using combined Canny and Laplacian operators.
    Args:
        img (torch.Tensor): Input image tensor [B, C, H, W]
    Returns:
        torch.Tensor: Combined edge map [B, 1, H, W]
    """
    # Convert to grayscale
    gray = kornia.color.rgb_to_grayscale(img).type(torch.cuda.FloatTensor)
    
    # Canny edge detection
    canny = kornia.filters.canny(gray.type(torch.cuda.FloatTensor), low_threshold=0.1, high_threshold=0.3)[0]
    
    # Laplacian edge detection
    laplacian = kornia.filters.laplacian(gray, kernel_size=3)
    
    # Combine and normalize
    return torch.clamp(canny + laplacian, 0, 1)
    
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
        #print(Y)
        """
        tensor([[  0.8911,  -0.2306,  40.9062,   0.2241,   0.8662, -31.1875],
        [  1.0146,  -0.3352,  62.1250,   0.2703,   0.8184, -28.0625],
        [  1.1250,  -0.3025,  21.2031,   0.2834,   1.0557, -62.3750],
        [  0.9136,  -0.3022,  44.1562,   0.2996,   0.9058, -39.7500]],
        """

        # Adversarial ground truths
        valid_ones = Variable(HalfTensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
        valid = valid_ones.fill_(0.9)
        fake = Variable(HalfTensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)

        # ------------------
        #  Train Generator
        # ------------------
        # Gen1/Disc1 == B, 
        # Gen2/Disc2 = A
        
        optimizer_G.zero_grad()
        print("+ + + optimizer_G.zero_grad() + + + ")
        with autocast():         
            # Cast real_A and real_B into a set of canny edges or laplacaian edges
            edge_A, edge_B = edge_detection(real_A), edge_detection(real_B)
            
            fake_A = generator2(real_B)
            fake_B = generator1(edge_A.expand(-1, 3, -1, -1))
            #warped_B, theta = model(img_A=fake_A, img_B=real_A, src=real_B)  
            warped_B, theta = model(img_A=real_A, img_B=edge_A.expand(-1, 3, -1, -1), src=real_B)  

            # min recon loss
            recon_loss = global_pixel_loss(fake_B, warped_B)

            # perceptual loss, LPIPS
            perc_A = criterion_lpips(real_A, fake_A)
            perc_weight = get_perceptual_weight(epoch, opt.n_epochs)
            perc_loss = perc_A.mean() * perc_weight

            # Replace tie_error with LoFTR feature loss
            #loftr_loss = loftr_feature_loss(warped_B2, real_A, epoch, max_epochs=opt.n_epochs)

            # FFT loss between fake_B and warped_B
            Amp_w, Pha_w = fft_components(warped_B)
            Amp_r, Pha_r = fft_components(fake_B)
            loss_Amp = criterion_amp(Amp_w, Amp_r)
            loss_Pha = criterion_phase(Pha_w, Pha_r)
            loss_FFT = (loss_Amp + loss_Pha).mean()

            Amp_w2, Pha_w2 = fft_components(real_A)
            Amp_r2, Pha_r2 = fft_components(fake_A)
            loss_Amp2 = criterion_amp(Amp_w2, Amp_r2)
            loss_Pha2 = criterion_phase(Pha_w2, Pha_r2)
            loss_FFT2 = (loss_Amp2 + loss_Pha2).mean()

            loss_FFT = (loss_FFT + loss_FFT2).mean()

            # Convert theta to tiepoints and compare with ground truth
            pred_source, pred_target = affine_to_tiepoints(theta.view(-1, 2, 3)) #theta
            gt_source, gt_target = affine_to_tiepoints(Y.view(-1, 2, 3)) #Y

            """
            print(" PRED SOURCE: theta ")
            print("theta: {}", pred_source)
            print("theta pred target: {}", pred_target)
            print("-----------")
            print(" GT SOURCE: Y ")
            print("y: {}", gt_source)
            print("y gt target: {}", gt_target)
            print("-----------")
            """
            
            # Calculate tiepoint loss using the transformed points (first arg - prediction, second arg - ground truth)
            # is this what Ivan meant as inverting? 
            tie_loss = F.smooth_l1_loss(gt_target, pred_target)
            print(tie_loss)
            

            # Adverarial ~ mode A goes to Discirminator 2
            loss_GAN1 = global_gen_loss(real_A, warped_B, fake_B, mode='B') # (A,WB), (A, FB)
            loss_GAN2 = global_gen_loss(real_A, warped_B, fake_A, mode='A') # (A, WB), (FA, WB) ~ GENERATOR 2 MAKES FAKE_A
            loss_GAN = (loss_GAN1 + loss_GAN2).mean()

            # Total Loss
            alpha1 = 0.001
            alpha2 = 0.25
            loss_G = (loss_GAN + perc_loss + recon_loss + loss_FFT*0.75 + tie_loss).mean()
            #loss_M = loftr_feature_loss(warped_B2, real_A.detach(), epoch, max_epochs=opt.n_epochs)

        scaler.scale(loss_G).backward()
        #scaler.scale(loss_M).backward()
        
        scaler.step(optimizer_G)
        #scaler.step(optimizer_M)
        print("+ + + optimizer_G.step() + + + ")
        #print("+ + + optimizer_M.step() + + + ")

        # -----------------------
        #  Train Discriminator
        # -----------------------

        optimizer_D.zero_grad()
        print("+ + + optimizer_D.zero_grad() + + +")
        with autocast():
            # Gen1/Disc1 == B, 
            # Gen2/Disc2 = A
            loss_D1 = global_disc_loss(real_A, warped_B.detach(), fake_img=fake_B, mode='B')
            loss_D2 = global_disc_loss(real_A, warped_B.detach(), fake_img=fake_A, mode='A') #only using d2
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
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, R(L1): %f, Perc(L1): %f, ] ETA: %s"
            % (
                epoch, #%d
                opt.n_epochs, #%d
                i, #%d
                len(dataloader), #%d
                loss_D.item(), #%f
                loss_G.item(), #%f - total G loss
                recon_loss.item(),
                perc_loss.item(),
                #loftr_loss.item(), LoFTR: %f
                time_left, #%s
            )
        )

        f.write(
           "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, R(L1): %f, Perc(L1): %f] ETA: %s"
            % (
                epoch, #%d
                opt.n_epochs, #%d
                i, #%d
                len(dataloader), #%d
                loss_D.item(), #%f
                loss_G.item(), #%f - total G loss
                recon_loss.item(),
                perc_loss.item(),
                #loftr_loss.item(),
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
