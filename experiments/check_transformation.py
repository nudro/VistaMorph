import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

def load_image(image_path):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img = Image.open(image_path)
    img = transform(img)
    return img.unsqueeze(0)  # Add batch dimension

def apply_transformation(img, theta):
    theta = theta.view(-1, 2, 3)
    grid = F.affine_grid(theta, img.size(), align_corners=True)
    output = F.grid_sample(img, grid, mode='bicubic', padding_mode='reflection', align_corners=True)
    return output, grid

def compute_inverse_theta(theta):
    a, b, tx, c, d, ty = theta
    det = a * d - b * c
    if det == 0:
        raise ValueError("Matrix is not invertible")
    inv_theta = torch.tensor([d / det, -b / det, -tx / det, -c / det, a / det, -ty / det], dtype=torch.float32)
    return inv_theta

def visualize_grid(grid, img_size):
    grid_np = grid.squeeze(0).cpu().numpy()
    plt.figure(figsize=(10, 10))
    plt.scatter(grid_np[:, :, 0], grid_np[:, :, 1], c='blue', s=1)
    plt.xlim(0, img_size[1])
    plt.ylim(0, img_size[0])
    plt.title('Transformation Grid')
    plt.savefig('grid_visualization.png')
    plt.close()

def visualize_transformation(optical, sar, theta):
    transformed, grid = apply_transformation(optical, theta)
    optical_np = optical.squeeze(0).permute(1, 2, 0).cpu().numpy()
    sar_np = sar.squeeze(0).permute(1, 2, 0).cpu().numpy()
    transformed_np = transformed.squeeze(0).permute(1, 2, 0).cpu().numpy()
    optical_np = (optical_np * 0.5 + 0.5) * 255
    sar_np = (sar_np * 0.5 + 0.5) * 255
    transformed_np = (transformed_np * 0.5 + 0.5) * 255
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(optical_np.astype(np.uint8))
    plt.title('Optical Image (Input)')
    plt.axis('off')
    plt.subplot(132)
    plt.imshow(transformed_np.astype(np.uint8))
    plt.title('Transformed Optical')
    plt.axis('off')
    plt.subplot(133)
    plt.imshow(sar_np.astype(np.uint8))
    plt.title('SAR Image (Target)')
    plt.axis('off')
    plt.savefig('transformation_check.png')
    plt.close()
    visualize_grid(grid, optical.size()[2:])

def main():
    img_path = '/Users/catherineordun/Documents/AI_Coding/VMorph_Assistant/VistaMorph/experiments/00131.png'
    img = load_image(img_path)
    optical = img[:, :, :, :256]
    sar = img[:, :, :, 256:]
    # Load the transformation matrix from the .txt file
    with open('/Users/catherineordun/Documents/AI_Coding/VMorph_Assistant/VistaMorph/experiments/00131.txt', 'r') as f:
        theta_values = list(map(float, f.read().strip().split()))
    # Extract the first two rows (6 values) for the affine transformation
    theta = torch.tensor(theta_values[:6], dtype=torch.float32)
    # Compute and apply inverse transformation first
    inv_theta = compute_inverse_theta(theta)
    visualize_transformation(sar, optical, inv_theta)
    # Then apply the original transformation
    visualize_transformation(optical, sar, theta)

if __name__ == "__main__":
    main() 