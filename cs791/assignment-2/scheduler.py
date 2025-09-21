import torch


class NoiseSchedulerDDPM:
    """
    Noise scheduler for the DDPM model

    Args:
        num_timesteps: int, the number of timesteps
        type: str, the type of scheduler to use
        **kwargs: additional arguments for the scheduler

    This object sets up all the constants like alpha, beta, sigma, etc. required for the DDPM model
    """

    def __init__(self, num_timesteps=50, type="linear", **kwargs):
        self.num_timesteps = num_timesteps
        self.type = type

        if type == "linear":
            self.init_linear_schedule(**kwargs)
        else:
            raise NotImplementedError(
                f"{type} scheduler is not implemented"
            )  # change this if you implement additional schedulers

    def init_linear_schedule(self, beta_start, beta_end):
        """
        Precompute whatever quantities are required for training and sampling
        """

        self.betas = torch.linspace(
            beta_start, beta_end, self.num_timesteps, dtype=torch.float32
        )

        self.alphas = None

    def __len__(self):
        return self.num_timesteps


class MaskSchedulerD3PM:
    """
    Mask scheduler for Discrete Diffusion (D3PM) models.

    Args:
        num_timesteps: int, number of timesteps in the diffusion process
        mask_type: str, type of mask scheduling ("uniform", "linear", etc.)
        **kwargs: additional arguments for mask scheduling

    This object sets up the mask schedule for each timestep.
    """

    def __init__(self, num_timesteps=50, mask_type="linear", **kwargs):
        self.num_timesteps = num_timesteps
        self.mask_type = mask_type

        if mask_type == "linear":
            self.init_linear_schedule(**kwargs)
        elif mask_type == "cosine":
            self.init_cosine_schedule(**kwargs)
        else:
            raise NotImplementedError(f"{mask_type} mask scheduler is not implemented")

    def init_linear_schedule(self, mask_prob_start=0.0, mask_prob_end=1.0, device=None):
        """
        Initializes a linear mask schedule where the mask probability increases linearly.
        """
        self.mask_probs = torch.linspace(
            mask_prob_start, mask_prob_end, self.num_timesteps, device=device
        )

    def init_cosine_schedule(self, mask_prob_start=0.0, mask_prob_end=1.0, device=None):
        """
        Initializes a cosine mask schedule where the mask probability follows a cosine curve.
        """

        s = 0.008  # offset for numerical stability
        steps = self.num_timesteps + 1
        x = torch.linspace(0, 1, steps)
        alphas_cumprod = torch.cos(((x + s) / (1 + s)) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        self.mask_probs = torch.clamp(betas, 0.0001, 0.9999).to(device)

    def add_noise(self, x, timestep, num_classes=10):
        """
        Add noise to discrete data according to the mask schedule.

        Args:
            x: Input tensor of shape (batch_size, 1, height, width) with values in [0, num_classes-1]
            timestep: Timestep tensor of shape (batch_size,)
            num_classes: Number of classes (including absorbing state)

        Returns:
            noisy_x: Noisy tensor with same shape as x
        """
        batch_size = x.shape[0]
        device = x.device

        # Get mask probabilities for this timestep
        mask_probs = self.mask_probs[timestep]

        # Create random mask
        mask = (
            torch.rand(batch_size, 1, x.shape[2], x.shape[3], device=device)
            < mask_probs[:, None, None, None]
        )

        # Set masked pixels to absorbing state (num_classes - 1)
        noisy_x = x.clone()
        noisy_x[mask] = num_classes - 1

        return noisy_x

    def __len__(self):
        return self.num_timesteps
