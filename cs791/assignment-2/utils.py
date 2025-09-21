import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchmetrics.image.fid import FrechetInceptionDistance


def discretize_mnist(x, num_classes=10):
    """
    Convert MNIST images to discrete values.
    Args:
        x: Tensor of shape (batch_size, 1, 28, 28) with values in [0, 1]
        num_classes: Number of discrete classes (including absorbing state)
    Returns:
        discrete_x: Tensor with values in [0, num_classes-2] (excluding absorbing state)
    """
    # Scale to [0, num_classes-2] and round to nearest integer
    discrete_x = torch.round(x * (num_classes - 2)).long()
    return discrete_x


def visualize_samples(samples, filename=None):
    # Assuming 'samples' is a tensor of generated images (batch_size, 1, 28, 28)
    if samples.is_cuda:
        samples = samples.cpu()

    num_samples = samples.shape[0]

    # Make a grid of images
    grid_img = torchvision.utils.make_grid(samples, nrow=int(num_samples**0.5))

    # Plot the grid
    plt.figure(figsize=(8, 8))
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.axis("off")
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()


def seed_everything(seed):
    # https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial3/Activation_Functions.html
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    random.seed(seed)


def compute_fid(real_images, fake_images):
    """
    Args:
        real_images (Actual images from the dataset): torch.Tensor, shape (N, 1, 28, 28), range [0, 1]
        fake_images (Generated images by the diffusion model): torch.Tensor, shape (N, 1, 28, 28), range [0, 1]
    Returns:
        fid (Frechet Inception Distance): torch.Tensor
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fid = FrechetInceptionDistance().to(device)
    # Move images to device
    real_images = real_images.to(device)
    fake_images = fake_images.to(device)
    # Convert to uint8 and [0, 255]
    real_images = (real_images * 255).clamp(0, 255).to(torch.uint8)
    fake_images = (fake_images * 255).clamp(0, 255).to(torch.uint8)
    # Convert 1 channel to 3 channels
    real_images = real_images.repeat(1, 3, 1, 1)
    fake_images = fake_images.repeat(1, 3, 1, 1)
    # Resize to 299x299
    real_images = F.interpolate(
        real_images.float(), size=(299, 299), mode="nearest"
    ).to(device)
    fake_images = F.interpolate(
        fake_images.float(), size=(299, 299), mode="nearest"
    ).to(device)
    # Convert back to uint8
    real_images = real_images.clamp(0, 255).to(torch.uint8)
    fake_images = fake_images.clamp(0, 255).to(torch.uint8)
    # Update FID metric
    fid.update(real_images, real=True)
    fid.update(fake_images, real=False)
    return fid.compute()
