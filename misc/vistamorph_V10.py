"""
VistaMorph V10: Registration with Cycle Consistency

This version implements registration with cycle consistency:
1. Maintains cycle consistency and identity preservation
2. Combines affine transformation with cycle consistency
3. Enhanced registration process with FFT loss

Key changes from V9:
- Removed feature extraction network
- Simplified registration process
- Enhanced cycle consistency
- Added FFT loss for better frequency domain alignment
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
            K.VisionTransformer(image_size=self.h, patch_size=64, in_channels=2)
        )

    def forward(self, x):
        with autocast():
            # Get ViT output [batch, 17, 768]
            out = self.vit(x).type(HalfTensor)
            # Reshape to match expected dimensions
            out = out.view(out.size(0), 1, 17, 768)
            print("out Vit:", out.size())
        return out

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
            # Ensure edge-detected images are properly concatenated
            img_input = torch.cat((img_A, img_B), 1)  # [batch, 2, H, W]
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
criterion_amp = nn.L1Loss()   # For FFT amplitude loss
criterion_phase = nn.L1Loss() # For FFT phase loss

############################
#  Utils
############################

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
    
    warped_B, theta = model(img_A=edge_A, img_B=edge_B, src=real_B)
    fake_A1 = generator2(warped_B)
    edge_fake_A1 = edge_detection(fake_A1)
    
    # Denormalize images
    def denormalize(x):
        return (x + 1) / 2.0
    
    real_A_denorm = denormalize(real_A.data)
    real_B_denorm = denormalize(real_B.data)
    warped_B_denorm = denormalize(warped_B.data)
    fake_A1_denorm = denormalize(fake_A1.data)
    edge_A_denorm = denormalize(edge_A.data)
    edge_fake_A1_denorm = denormalize(edge_fake_A1.data)
    
    # Stack images
    img_sample_global = torch.cat((
        real_A_denorm, real_B_denorm, warped_B_denorm, 
        fake_A1_denorm, edge_A_denorm, edge_fake_A1_denorm
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

def global_pixel_loss(real_B, fake_B):
    loss_pix = criterion_L1(fake_B, real_B)
    return loss_pix

def global_gen_loss(real_A, real_B, fake_img, mode):
    if mode=='A':
        pred_fake = discriminator2(fake_img, real_B)
        real_pred = discriminator2(real_A, real_B)
        loss_GAN = criterion_GAN(pred_fake - real_pred.detach(), valid)
    return loss_GAN

def global_disc_loss(real_A, real_B, fake_img, mode):
    if mode=='A':
        pred_real = discriminator2(real_A, real_B)
        pred_fake = discriminator2(fake_img.detach(), real_B)
        loss_real = criterion_GAN(pred_real - pred_fake, valid)
        loss_fake = criterion_GAN(pred_fake - pred_real, fake)
        loss_D = 0.25*(loss_real + loss_fake)
    return loss_D

def geometric_tie_loss(pred_theta, target_theta):
    """
    Sophisticated tie-loss that combines geometric properties and smooth L1 loss.
    Args:
        pred_theta: Predicted transformation matrix [B, 2, 3]
        target_theta: Target transformation matrix [B, 2, 3]
    Returns:
        Combined loss value
    """
    # Extract rotation and scaling components
    pred_rot = torch.atan2(pred_theta[:, 1, 0], pred_theta[:, 0, 0])
    target_rot = torch.atan2(target_theta[:, 1, 0], target_theta[:, 0, 0])
    
    # Calculate rotation loss using cosine similarity
    rot_loss = 1 - torch.cos(pred_rot - target_rot)
    
    # Calculate scaling loss
    pred_scale = torch.sqrt(pred_theta[:, 0, 0]**2 + pred_theta[:, 1, 0]**2)
    target_scale = torch.sqrt(target_theta[:, 0, 0]**2 + target_theta[:, 1, 0]**2)
    scale_loss = F.smooth_l1_loss(pred_scale, target_scale)
    
    # Calculate translation loss
    trans_loss = F.smooth_l1_loss(pred_theta[:, :, 2], target_theta[:, :, 2])
    
    # Calculate determinant loss to preserve area
    pred_det = pred_theta[:, 0, 0] * pred_theta[:, 1, 1] - pred_theta[:, 0, 1] * pred_theta[:, 1, 0]
    target_det = target_theta[:, 0, 0] * target_theta[:, 1, 1] - target_theta[:, 0, 1] * target_theta[:, 1, 0]
    det_loss = F.smooth_l1_loss(pred_det, target_det)
    
    # Combine losses with weights
    total_loss = (0.3 * rot_loss + 0.3 * scale_loss + 0.2 * trans_loss + 0.2 * det_loss).mean()
    
    return total_loss

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
generator2 = GeneratorUNet2(input_shape_patch) # for fake_A
discriminator2 = Discriminator2(input_shape_patch) # for fake_A

if cuda:
    generator2 = generator2.cuda()
    discriminator2 = discriminator2.cuda()
    model = Net().cuda()

    criterion_GAN.cuda()
    criterion_lpips.cuda()
    criterion_L1.cuda()
    criterion_MSE.cuda()
    criterion_amp.cuda()
    criterion_phase.cuda()

# Trained on multigpus - change your device ids if needed
generator2 = torch.nn.DataParallel(generator2, device_ids=[0,1])
discriminator2 = torch.nn.DataParallel(discriminator2, device_ids=[0,1])
model = torch.nn.DataParallel(model, device_ids=[0,1])


if opt.epoch != 0:
    # Load pretrained models
    generator2.load_state_dict(torch.load("./saved_models/%s/generator2_%d.pth" % (opt.experiment, opt.epoch)))
    discriminator2.load_state_dict(torch.load("./saved_models/%s/discriminator2_%d.pth" % (opt.experiment, opt.epoch)))
    model.load_state_dict(torch.load("./saved_models/%s/net_%d.pth" % (opt.experiment, opt.epoch)))

else:
    # Initialize weights
    generator2.apply(weights_init_normal)
    discriminator2.apply(weights_init_normal)
    model.apply(weights_init_normal)

# Optimizers - Jointly train generators and STN
optimizer_G = torch.optim.Adam(itertools.chain(generator2.parameters(), model.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator2.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

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
            # Cast real_A and real_B into a set of canny edges or laplacaian edges
            edge_A, edge_B = edge_detection(real_A), edge_detection(real_B)
            
            # Spatial transformation to align real_B with real_A
            warped_B, theta = model(img_A=edge_A, img_B=edge_B, src=real_B)
            
            # Generate fake_A1 from warped_B - this should maintain real_A features
            fake_A1 = generator2(warped_B)
            edge_fake_A1 = edge_detection(fake_A1)

            # Reconstruction loss between edge_fake_A1 and edge_A
            recon_loss = global_pixel_loss(edge_fake_A1, edge_A)

            # Identity loss to maintain real_A features in fake_A1
            identity_loss = global_pixel_loss(fake_A1, real_A)

            # FFT loss between real_A and warped_B
            Amp_w, Pha_w = fft_components(warped_B)
            Amp_r, Pha_r = fft_components(real_A)
            loss_Amp = criterion_amp(Amp_w, Amp_r)
            loss_Pha = criterion_phase(Pha_w, Pha_r)
            loss_FFT = (loss_Amp + loss_Pha).mean()

            # Adversarial losses
            loss_GAN = global_gen_loss(real_A, real_B, fake_img=fake_A1, mode='A')

            # Replace the tie_loss calculation with the new geometric loss
            tie_loss = geometric_tie_loss(theta.view(-1, 2, 3), Y.view(-1, 2, 3))

            # Total Loss with adjusted weights
            alpha1 = 0.3   # Reconstruction loss weight
            alpha2 = 0.25  # Tie point loss weight
            alpha3 = 0.15  # Identity loss weight
            alpha4 = 0.15  # FFT loss weight
            alpha5 = 0.15  # GAN loss weight
            
            loss_G = (alpha5 * loss_GAN + 
                     alpha1 * recon_loss + 
                     alpha2 * tie_loss + 
                     alpha3 * identity_loss +
                     alpha4 * loss_FFT).mean()

            # Backward pass
            scaler.scale(loss_G).backward()
            scaler.step(optimizer_G)
            print("+ + + optimizer_G.step() + + + ")

            # -----------------------
            #  Train Discriminator
            # -----------------------

            optimizer_D.zero_grad()
            print("+ + + optimizer_D.zero_grad() + + +")
            with autocast():
                loss_D = global_disc_loss(real_A, real_B, fake_img=fake_A1, mode='A')

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
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, R: %f, T: %f, I: %f, F: %f] ETA: %s"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    loss_D.item(),
                    loss_G.item(),
                    recon_loss.item(),
                    tie_loss.item(),
                    identity_loss.item(),
                    loss_FFT.item(),
                    time_left,
                )
            )

            f.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, R: %f, T: %f, I: %f, F: %f] ETA: %s"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    loss_D.item(),
                    loss_G.item(),
                    recon_loss.item(),
                    tie_loss.item(),
                    identity_loss.item(),
                    loss_FFT.item(),
                    time_left,
                )
            )

            # If at sample interval save image
            if batches_done % opt.sample_interval == 0:
                sample_images(batches_done)

        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(generator2.state_dict(), "./saved_models/%s/generator2_%d.pth" % (opt.experiment, epoch))
            torch.save(discriminator2.state_dict(), "./saved_models/%s/discriminator2_%d.pth" % (opt.experiment, epoch))
            torch.save(model.state_dict(), "./saved_models/%s/net_%d.pth" % (opt.experiment, epoch))

        