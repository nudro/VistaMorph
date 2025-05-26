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
import kornia.contrib as K
import matplotlib.pyplot as plt
#from pynvml import *
from transformers import TrainingArguments, Trainer, logging
from accelerate import Accelerator
from diffusers import DDPMScheduler, UNet2DModel



"""
V3 - 2 Stacked STNs with Grid Regularization and Diffusion Model
Params: 
Rs = F.grid_sample(src_tensors[j], rs_grid, mode='nearest', padding_mode='border', align_corners=False)
NO GAN
"""


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
parser.add_argument("--sample_interval", type=int, default=30, help="interval between sampling of images from generators")
parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between model checkpoints")
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
criterion_L1 = nn.L1Loss()
loss_fn = nn.MSELoss()

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


def sample_images(batches_done):
    imgs = next(iter(test_dataloader)) # batch_size = 1
    real_A = Variable(imgs["A"].type(HalfTensor)).cuda() # torch.Size([1, 3, 256, 256])
    real_B = Variable(imgs["B"].type(HalfTensor)).cuda()
    Y = Variable(imgs["Y"].type(HalfTensor)).cuda()  # Get Y from the test dataloader
    
    noise = torch.randn_like(real_A).cuda() 
    timesteps = torch.randint(0, 999, (real_B.shape[0],)).long().cuda()
    noisy_A = noise_scheduler.add_noise(real_A, noise, timesteps)
    pred = Diff(noisy_A, timesteps, Y)  # Get the denoised output
    warped_B, theta = model(img_A=noisy_A, img_B=real_B, src=real_B) 

    # Save the regular image comparison
    img_sample_global = torch.cat((real_A.data, real_B.data, noisy_A.data, pred.data, warped_B.data), -1)
    save_image(img_sample_global, "./images/%s/%s.png" % (opt.experiment, batches_done), nrow=4, normalize=True)

##################
# DDPM
##################
class ClassConditionedUnet(nn.Module):
    
    def __init__(self, num_classes=4, class_emb_size=4):
        super().__init__()
        # Replace embedding with a linear layer for continuous values
        self.class_cond = nn.Linear(6, class_emb_size)  # 6 for affine matrix values

        # Self.model is an unconditional UNet with extra input channels to accept the conditioning information
        self.model = UNet2DModel(
            sample_size=128,            # the target image resolution
            in_channels=3 + class_emb_size, # Additional input channels for class cond.
            out_channels=3,           # the number of output channels
            layers_per_block=1,       # how many ResNet layers to use per UNet block
            block_out_channels=(32, 64, 64), # More channels -> more parameters
            down_block_types=( 
                "DownBlock2D",        # a regular ResNet downsampling block
                "AttnDownBlock2D",    # a ResNet downsampling block with spatial self-attention
                "AttnDownBlock2D",
            ), 
            up_block_types=(
                "AttnUpBlock2D", 
                "AttnUpBlock2D",      # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",          # a regular ResNet upsampling block
              ),
        )
        
    def forward(self, x, t, class_labels):
        with autocast(): 
            bs, ch, w, h = x.shape  
            # Process continuous values through linear layer
            class_cond = self.class_cond(class_labels.view(bs, -1))  # Reshape to [batch_size, 6]
            class_cond = class_cond.view(bs, -1, 1, 1).expand(bs, -1, w, h)
            net_input = torch.cat((x, class_cond), 1)          
            output = (self.model(net_input, t).sample)
    
        return output

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

        # nn.Linear(1*257*768, 1024), # (hard-coded in, 257, 768 based on ViT output)
        self.fc_loc = nn.Sequential(
            nn.Linear(1*17*768, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512), # Added more layers, will this stop the affine warp?
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.Sigmoid(), # <--- SIGMOID will this force the matrix values between [-1,+1]
            nn.Linear(256, 3*2))
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
            img_input = torch.cat((img_A, img_B), 1)
            dtheta = self.stn_phi(img_input)
            identity_theta = torch.tensor(identity_matrix, dtype=torch.float).cuda()
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
                # Adjusting some of the parameters like align_cornrs, mode, and padding_mode will change output
                # Based on my experiments, the below are the best configurations
                rs_grid = F.affine_grid(theta_batches[i], src_tensors[i].size(), align_corners=True)

                #===============
                # add regularization
                grid_displacement = torch.norm(rs_grid - rs_grid.mean(dim=[1,2], keepdim=True), dim=-1)
                
                # Adaptive noise scaling based on displacement
                noise_scale = torch.clamp(0.02 * grid_displacement, max=0.05)
                noise = torch.randn_like(rs_grid) * noise_scale.unsqueeze(-1)
                rs_grid = rs_grid + noise
                #-----------------
                
                Rs = F.grid_sample(src_tensors[j], rs_grid, mode='nearest', padding_mode='border', align_corners=False)
                warped.append(Rs.type(HalfTensor))

        return torch.cat(warped)


class StackedSTN(nn.Module):
    def __init__(self, num_stns=3):
        super(StackedSTN, self).__init__()
        self.num_stns = num_stns
        self.stns = nn.ModuleList([Net() for _ in range(num_stns)])
        
    def compose_transformations(self, theta1, theta2):
        """Compose two affine transformations."""
        # Convert to homogeneous coordinates
        theta1_h = torch.cat([theta1, torch.tensor([[0, 0, 1]], device=theta1.device)], dim=0)
        theta2_h = torch.cat([theta2, torch.tensor([[0, 0, 1]], device=theta2.device)], dim=0)
        
        # Matrix multiplication
        composed = torch.matmul(theta2_h, theta1_h)
        
        # Return only the affine part
        return composed[:2, :]
    
    def forward(self, img_A, img_B, src):
        identity_matrix = [1, 0, 0, 0, 1, 0]
        identity_theta = torch.tensor(identity_matrix, dtype=torch.float).cuda()
        
        with autocast():
            current_src = src
            total_theta = None
            
            for i, stn in enumerate(self.stns):
                # Get transformation from current STN
                if i == 0:
                    # First STN uses original images
                    img_input = torch.cat((img_A, img_B), 1)
                else:
                    # Subsequent STNs use warped image from previous stage
                    img_input = torch.cat((img_A, current_src), 1)
                
                # Get transformation parameters
                dtheta = stn.stn_phi(img_input)
                dtheta = dtheta.reshape(img_A.size(0), 2*3)
                dtheta = dtheta + identity_theta.unsqueeze(0).repeat(img_A.size(0), 1)
                
                # Update total transformation
                if total_theta is None:
                    total_theta = dtheta
                else:
                    # Compose transformations
                    batch_thetas = []
                    for b in range(dtheta.size(0)):
                        theta1 = total_theta[b].view(2, 3)
                        theta2 = dtheta[b].view(2, 3)
                        composed = self.compose_transformations(theta1, theta2)
                        batch_thetas.append(composed.view(-1))
                    total_theta = torch.stack(batch_thetas)
                
                # Apply current transformation
                theta_batches = []
                for t in dtheta:
                    this_theta = (t.view(-1, 2, 3)).reshape(1, 2, 3)
                    theta_batches.append(this_theta)
                
                src_tensors = []
                for img in current_src:
                    this_img_src = img.reshape(1, img.size(0), img.size(1), img.size(2))
                    src_tensors.append(this_img_src)
                
                # Apply transformation
                warped = []
                for j in range(len(src_tensors)):
                    rs_grid = F.affine_grid(theta_batches[j], src_tensors[j].size(), align_corners=True)
                    Rs = F.grid_sample(src_tensors[j], rs_grid, mode='bicubic', padding_mode='border', align_corners=True)
                    warped.append(Rs.type(HalfTensor))
                
                current_src = torch.cat(warped)
            
            return current_src, total_theta


# ===========================================================
# Initialize generator and discriminator
# ===========================================================

input_shape_patch = (opt.channels, opt.img_height, opt.img_width)

if cuda:
    model = StackedSTN(num_stns=2).cuda() #<-------- STACK
    criterion_L1.cuda()
    Diff = ClassConditionedUnet().cuda()

# Trained on multigpus - change your device ids if needed
model = torch.nn.DataParallel(model, device_ids=[0, 1])
Diff = torch.nn.DataParallel(Diff, device_ids=[0,1])

if opt.epoch != 0:
    model.load_state_dict(torch.load("./saved_models/%s/stn_%d.pth" % (opt.experiment, opt.epoch)))
else:
    model.apply(weights_init_normal)

# Optimizers - Jointly train diffusion and STN
optimizer_M = torch.optim.Adam(itertools.chain(model.parameters(), Diff.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2))

# Tensor type - only use HalfTensor in this AMP script
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
HalfTensor = torch.cuda.HalfTensor if cuda else torch.HalfTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


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

# Scheduler
noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')

# AMP
scaler = GradScaler()

for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        real_A = Variable(batch["A"].type(HalfTensor)).cuda()
        real_B = Variable(batch["B"].type(HalfTensor)).cuda()
        Y = Variable(batch["Y"].type(HalfTensor)).cuda()  # Ground truth affine matrix

        # Reshape Y to match the number of channels in real_A
        Y = Y.view(Y.size(0), 6, 1, 1)  # Reshape to (batch_size, 6, 1, 1)
        Y = Y.expand(-1, 6, -1, -1)  # Expand Y to match the number of channels in real_A

        # ------------------
        #  Train Stacked STN
        # ------------------

        optimizer_M.zero_grad()
        print("+ + + optimizer_M.zero_grad() + + + ")
        with autocast():  

            noise = torch.randn_like(real_A) 
            timesteps = torch.randint(0, 999, (real_B.shape[0],)).long().cuda()

            # Debug prints
            print("real_A shape:", real_A.shape)
            print("noise shape:", noise.shape)
            print("Y shape:", Y.shape)
            print("timesteps shape:", timesteps.shape)

            # Use noise instead of Y for add_noise
            noisy_A = noise_scheduler.add_noise(real_A, noise, timesteps)

            # No need to convert Y to long since we're using continuous values
            pred = Diff(noisy_A, timesteps, Y)  # Pass Y directly

            # noise loss
            loss_noise = (loss_fn(pred, noise)).mean()

            # STN
            warped_B, theta = model(img_A=noisy_A, img_B=real_B, src=real_B) 

            recon_loss = criterion_L1(warped_B, noisy_A) 
            
            # Convert theta to tiepoints and compare with ground truth
            pred_source, pred_target = affine_to_tiepoints(theta.view(-1, 2, 3)) #theta
            gt_source, gt_target = affine_to_tiepoints(Y.view(-1, 2, 3)) #Y
            tie_loss = F.smooth_l1_loss(gt_target, pred_target)

            # Total Loss
            loss_M = tie_loss + recon_loss + loss_noise
    
        scaler.scale(loss_M).backward()
        scaler.step(optimizer_M)
        print("+ + + optimizer_M.step() + + + ")
        scaler.update()

        # --------------
        #  Log Progress
        # --------------

        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [M loss: %f] [R(L1): %f, Tie: %f, ] ETA: %s"
            % (
                epoch, #%d
                opt.n_epochs, #%d
                i, #%d
                len(dataloader), #%d
                loss_M.item(), #%f
                recon_loss.item(),
                tie_loss.item(),
                time_left, #%s
            )
        )

        # If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done)


    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(model.state_dict(), "./saved_models/%s/stn_%d.pth" % (opt.experiment, epoch))
        torch.save(Net.state_dict(), "/home/local/AD/cordun1/experiments/TF-Diff/saved_models/prototype/Net_%d.pth" % epoch)
