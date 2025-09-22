import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)

        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x, time_emb):
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.relu(h)

        # Add time embedding
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None]

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.dropout(h)

        return h + self.residual_conv(x)


class UNet(nn.Module):
    def __init__(
        self, in_channels=1, out_channels=10, time_emb_dim=128, num_classes=None
    ):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.num_classes = num_classes

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
        )

        # Class embedding for conditional model
        if num_classes is not None:
            self.class_emb = nn.Embedding(num_classes, time_emb_dim)

        # Downsampling
        self.down1 = ResidualBlock(in_channels, 64, time_emb_dim)
        self.down2 = ResidualBlock(64, 128, time_emb_dim)
        self.down3 = ResidualBlock(128, 256, time_emb_dim)

        # Bottleneck
        self.bottleneck = ResidualBlock(256, 512, time_emb_dim)

        # Upsampling
        self.up3 = ResidualBlock(512 + 256, 256, time_emb_dim)
        self.up2 = ResidualBlock(256 + 128, 128, time_emb_dim)
        self.up1 = ResidualBlock(128 + 64, 64, time_emb_dim)

        # Output layer
        self.out = nn.Conv2d(64, out_channels, 1)

        # Pooling and upsampling
        self.downsample = nn.MaxPool2d(2)
        # Use ConvTranspose2d for upsampling to control output size
        self.upsample_bottleneck = nn.ConvTranspose2d(
            512, 512, kernel_size=3, stride=2, padding=0, output_padding=0
        )
        self.upsample_up3 = nn.ConvTranspose2d(
            256, 256, kernel_size=4, stride=2, padding=1
        )
        self.upsample_up2 = nn.ConvTranspose2d(
            128, 128, kernel_size=4, stride=2, padding=1
        )

    def forward(self, x, timestep, class_labels=None):
        # Time embedding
        t = self.time_mlp(timestep)

        if self.num_classes is not None and class_labels is not None:
            class_emb = self.class_emb(class_labels)
            t = t + class_emb

        # Downsampling
        d1 = self.down1(x, t)
        d2 = self.down2(self.downsample(d1), t)
        d3 = self.down3(self.downsample(d2), t)

        # Bottleneck
        bottleneck = self.bottleneck(self.downsample(d3), t)

        # Upsampling
        # Use ConvTranspose2d for upsampling
        u3 = self.up3(torch.cat([self.upsample_bottleneck(bottleneck), d3], dim=1), t)
        u2 = self.up2(torch.cat([self.upsample_up3(u3), d2], dim=1), t)
        u1 = self.up1(torch.cat([self.upsample_up2(u2), d1], dim=1), t)

        return self.out(u1)


class D3PM(nn.Module):
    def __init__(self, num_classes=10, time_emb_dim=128):
        super().__init__()
        self.num_classes = num_classes
        self.unet = UNet(
            in_channels=1,
            out_channels=num_classes,
            time_emb_dim=time_emb_dim,
            num_classes=num_classes,
        )

    def forward(self, x, timestep):
        """
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28) with values in [0, num_classes-1]
            timestep: Timestep tensor of shape (batch_size,)
        Returns:
            logits: Logits for each class at each pixel, shape (batch_size, num_classes, 28, 28)
        """
        return self.unet(x, timestep)


class ConditionalD3PM(nn.Module):
    def __init__(self, num_classes=10, time_emb_dim=128):
        super().__init__()
        self.num_classes = num_classes
        self.unet = UNet(
            in_channels=1,
            out_channels=num_classes,
            time_emb_dim=time_emb_dim,
            num_classes=num_classes,
        )

    def forward(self, x, timestep, class_labels):
        """
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28) with values in [0, num_classes-1]
            timestep: Timestep tensor of shape (batch_size,)
            class_labels: Class labels tensor of shape (batch_size,)
        Returns:
            logits: Logits for each class at each pixel, shape (batch_size, num_classes, 28, 28)
        """
        return self.unet(x, timestep, class_labels)


class DDPM(nn.Module):
    def __init__(self, time_emb_dim=128):
        super().__init__()
        # Unconditional DDPM predicts Gaussian noise (epsilon) with 1 output channel
        self.unet = UNet(
            in_channels=1, out_channels=1, time_emb_dim=time_emb_dim, num_classes=None
        )

    def forward(self, x, timestep):
        """
        Args:
            x: float tensor in [0, 1], shape (batch_size, 1, 28, 28)
            timestep: LongTensor shape (batch_size,)
        Returns:
            epsilon prediction: shape (batch_size, 1, 28, 28)
        """
        return self.unet(x, timestep)


class ConditionalDDPM(nn.Module):
    def __init__(self, num_classes, time_emb_dim=128):
        super().__init__()
        self.num_classes = num_classes
        # Conditional DDPM predicts epsilon with class-conditioning via embeddings
        self.unet = UNet(
            in_channels=1,
            out_channels=1,
            time_emb_dim=time_emb_dim,
            num_classes=num_classes,
        )

    def forward(self, x, timestep, class_labels):
        """
        Args:
            x: float tensor in [0, 1], shape (batch_size, 1, 28, 28)
            timestep: LongTensor shape (batch_size,)
            class_labels: LongTensor shape (batch_size,)
        Returns:
            epsilon prediction: shape (batch_size, 1, 28, 28)
        """
        return self.unet(x, timestep, class_labels)
