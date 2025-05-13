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
        # Change back to 6 channels since we'll expand edge images to 3 channels each
        self.vit = nn.Sequential(
            K.VisionTransformer(image_size=self.h, patch_size=64, in_channels=6)
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



##########################
# Loss functions
##########################
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

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def sample_images(batches_done, real_A, real_B, warped_B):
    """Save a sample of training images with keypoint matches visualization.
    Args:
        batches_done: Current batch number
        real_A: Real image tensor
        real_B: Real B image tensor
        warped_B: Warped B image tensor
    """
    def denormalize(x):
        return (x + 1) / 2.0
    
    # Denormalize images
    real_A_denorm = denormalize(real_A.data)
    real_B_denorm = denormalize(real_B.data)
    warped_B_denorm = denormalize(warped_B.data)
    
    # Get keypoint matches for visualization
    matches, kpts_warped, kpts_real = adalam(warped_B, real_A, warped_B.device, return_matches=True)
    
    # Create visualization of matches
    if matches is not None and len(matches) > 0:
        # Convert tensors to numpy for visualization
        real_A_np = real_A_denorm[0].cpu().numpy().transpose(1, 2, 0)
        real_B_np = real_B_denorm[0].cpu().numpy().transpose(1, 2, 0)
        warped_B_np = warped_B_denorm[0].cpu().numpy().transpose(1, 2, 0)
        
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
        
        plt.savefig(f"images/{opt.experiment}/matches_{batches_done}.png")
        plt.close()
    
    # Save original image comparison
    img_sample = torch.cat((
        real_A_denorm, real_B_denorm, warped_B_denorm
    ), -1)
    save_image(img_sample, f"images/{opt.experiment}/{batches_done}.png", nrow=3, normalize=False)

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
# Initialize model
# ===========================================================
input_shape_patch = (opt.channels, opt.img_height, opt.img_width)
model = Net()

if cuda:
    model = model.cuda()
    criterion_MSE.cuda()
    criterion_amp.cuda()
    criterion_phase.cuda()

# Trained on multigpus - change your device ids if needed
model = torch.nn.DataParallel(model, device_ids=[0,1])

if opt.epoch != 0:
    # Load pretrained models
    model.load_state_dict(torch.load("./saved_models/%s/net_%d.pth" % (opt.experiment, opt.epoch)))
else:
    # Initialize weights
    model.apply(weights_init_normal)

# Optimizers
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

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

def homog_loss(matches, kpts_warped, kpts_real, device):
    """Calculate homography loss from matches and keypoints.
    Args:
        matches: Matched keypoint indices
        kpts_warped: Keypoints from warped image
        kpts_real: Keypoints from real image
        device: Device to use for tensor operations
    Returns:
        torch.Tensor: Homography loss
    """
    if matches is None or len(matches) == 0:
        return torch.tensor(0.0, device=device)
        
    src_pts = kpts_warped[matches[:, 0]]
    dst_pts = kpts_real[matches[:, 1]]
    
    # Estimate homography using RANSAC
    H, inliers = kornia.geometry.find_homography_dlt(
        src_pts, dst_pts, weights=torch.ones(len(src_pts), device=device)
    )
    
    # Calculate reprojection error
    if H is not None:
        src_pts_h = kornia.geometry.convert_points_to_homogeneous(src_pts)
        dst_pts_proj = kornia.geometry.transform_points(H, src_pts)
        reproj_error = torch.norm(dst_pts - dst_pts_proj, dim=1)
        return reproj_error.mean()
    else:
        return torch.tensor(0.0, device=device)

def adalam(warped_B, real_A, device, return_matches=False):
    """Perform keypoint detection and matching using KeyNet and AdaLAM.
    Args:
        warped_B: Warped image tensor
        real_A: Real image tensor
        device: Device to use for tensor operations
        return_matches: Whether to return matches and keypoints for visualization
    Returns:
        If return_matches is False:
            torch.Tensor: Homography loss
        If return_matches is True:
            tuple: (matches, kpts_warped, kpts_real)
    """
    # Convert images to grayscale for keypoint detection
    warped_B_gray = kornia.color.rgb_to_grayscale(warped_B)
    real_A_gray = kornia.color.rgb_to_grayscale(real_A)
    
    # Detect keypoints using KeyNet
    keynet = kornia.feature.KeyNetDetector()
    kpts_warped, desc_warped = keynet(warped_B_gray)
    kpts_real, desc_real = keynet(real_A_gray)
    
    # Match keypoints using AdaLAM
    adalam_matcher = kornia.feature.AdaLAM()
    matches = adalam_matcher(desc_warped, desc_real, kpts_warped, kpts_real)
    
    if return_matches:
        return matches, kpts_warped, kpts_real
    
    # Calculate homography loss
    return homog_loss(matches, kpts_warped, kpts_real, device)

for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        real_A = Variable(batch["A"].type(HalfTensor))
        real_B = Variable(batch["B"].type(HalfTensor))
        Y = Variable(batch["Y"].type(HalfTensor))  # Ground truth affine matrix

        # ------------------
        #  Train Model
        # ------------------
        optimizer.zero_grad()
        print("+ + + optimizer.zero_grad() + + + ")
        with autocast():
            # Cast real_A and real_B into a set of canny edges or laplacaian edges
            edge_A, edge_B = edge_detection(real_A), edge_detection(real_B)
            
            # Spatial transformation to align real_B with real_A
            warped_B, theta = model(img_A=edge_A, img_B=edge_B, src=real_B)

            # FFT loss between real_A and warped_B
            Amp_w, Pha_w = fft_components(warped_B)
            Amp_r, Pha_r = fft_components(real_A)
            loss_Amp = criterion_amp(Amp_w, Amp_r)
            loss_Pha = criterion_phase(Pha_w, Pha_r)
            loss_FFT = (loss_Amp + loss_Pha).mean()

            # Calculate homography loss using AdaLAM
            homography_loss = adalam(warped_B, real_A, warped_B.device)

            # Convert theta to tiepoints and compare with ground truth
            pred_source, pred_target = affine_to_tiepoints(theta.view(-1, 2, 3))
            gt_source, gt_target = affine_to_tiepoints(Y.view(-1, 2, 3))
            
            # Calculate tiepoint loss using the transformed points
            tie_loss = F.smooth_l1_loss(pred_target, gt_target)

            # Total Loss with adjusted weights
            alpha1 = 0.6   # Tie point loss weight
            alpha2 = 0.3   # FFT loss weight
            alpha3 = 0.1   # Homography loss weight
            
            loss = (alpha1 * tie_loss + alpha2 * loss_FFT + alpha3 * homography_loss).mean()

            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            print("+ + + optimizer.step() + + + ")

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
                "\r[Epoch %d/%d] [Batch %d/%d] [Loss: %f, T: %f, F: %f, H: %f] ETA: %s"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    loss.item(),
                    tie_loss.item(),
                    loss_FFT.item(),
                    homography_loss.item(),
                    time_left,
                )
            )

            f.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [Loss: %f, T: %f, F: %f, H: %f] ETA: %s"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    loss.item(),
                    tie_loss.item(),
                    loss_FFT.item(),
                    homography_loss.item(),
                    time_left,
                )
            )

            # If at sample interval save image
            if batches_done % opt.sample_interval == 0:
                sample_images(batches_done, real_A, real_B, warped_B)

        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(model.state_dict(), "./saved_models/%s/net_%d.pth" % (opt.experiment, epoch))

        