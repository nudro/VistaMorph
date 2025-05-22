import torch
import kornia
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Function to create a binary mask from an image tensor
def create_binary_mask(img, threshold=0.0):
    if img.size(1) > 1:
        gray = kornia.color.rgb_to_grayscale(img)
    else:
        gray = img
    mask = (gray > threshold).float()
    return mask

# Load the image
img_path = 'experiments/04820.png'
img = Image.open(img_path)
img_np = np.array(img)

# Assume the image is [H, W*2, C] and split into optical and SAR
h, w, c = img_np.shape
mid = w // 2
optical_np = img_np[:, :mid, :]
sar_np = img_np[:, mid:, :]

# Convert to torch tensors and normalize to [-1, 1]
def to_tensor(img_np):
    t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
    t = t * 2 - 1
    return t.unsqueeze(0)  # [1, 3, H, W]

optical = to_tensor(optical_np)
sar = to_tensor(sar_np)

# Create binary masks
mask_optical = create_binary_mask(optical).expand(-1, 3, -1, -1)
mask_sar = create_binary_mask(sar).expand(-1, 3, -1, -1)

# Convert masks to numpy for visualization
mask_optical_np = mask_optical.squeeze(0).permute(1, 2, 0).cpu().numpy()
mask_sar_np = mask_sar.squeeze(0).permute(1, 2, 0).cpu().numpy()

# Convert original images for visualization
optical_vis = ((optical.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
sar_vis = ((sar.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1) / 2 * 255).astype(np.uint8)

# Plot
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs[0, 0].imshow(optical_vis)
axs[0, 0].set_title('Optical Image')
axs[0, 0].axis('off')
axs[0, 1].imshow(sar_vis)
axs[0, 1].set_title('SAR Image')
axs[0, 1].axis('off')
axs[1, 0].imshow(mask_optical_np, cmap='gray')
axs[1, 0].set_title('Optical Binary Mask')
axs[1, 0].axis('off')
axs[1, 1].imshow(mask_sar_np, cmap='gray')
axs[1, 1].set_title('SAR Binary Mask')
axs[1, 1].axis('off')
plt.tight_layout()
plt.savefig('experiments/04820_binary_mask_visualization.png')
plt.show() 