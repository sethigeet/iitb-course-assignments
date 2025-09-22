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
        elif type == "cosine":
            self.init_cosine_schedule(**kwargs)
        else:
            raise NotImplementedError(
                f"{type} scheduler is not implemented"
            )  # change this if you implement additional schedulers

    def init_linear_schedule(self, beta_start=1e-4, beta_end=2e-2, device=None):
        """
        Precompute quantities required for training and sampling following Ho et al. 2020.
        """

        betas = torch.linspace(
            beta_start, beta_end, self.num_timesteps, dtype=torch.float32, device=device
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [
                torch.tensor([1.0], dtype=torch.float32, device=betas.device),
                alphas_cumprod[:-1],
            ],
            dim=0,
        )

        # Store buffers
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev

        # Useful terms
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # Posterior variance for q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        # Clamp for numerical stability
        self.posterior_variance = torch.clamp(self.posterior_variance, min=1e-20)

    def init_cosine_schedule(self, s: float = 0.008, device=None):
        """Cosine schedule from Nichol & Dhariwal 2021 (improved DDPM)."""
        steps = self.num_timesteps + 1
        x = torch.linspace(0, 1, steps, device=device)
        alphas_cumprod = torch.cos(((x + s) / (1 + s)) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clamp(betas, 1e-8, 0.999)

        # Reuse linear init path to set derived terms
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [
                torch.tensor([1.0], dtype=torch.float32, device=betas.device),
                alphas_cumprod[:-1],
            ],
            dim=0,
        )

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev

        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_variance = torch.clamp(self.posterior_variance, min=1e-20)

    def add_noise(self, x0, timesteps, noise=None):
        """
        q_sample: Diffuse the data by adding noise according to q(x_t | x_0).

        Args:
            x0: clean input in [-1, 1]
            timesteps: LongTensor of shape (batch_size,)
            noise: optional pre-sampled noise matching x0

        Returns:
            x_t with the same shape as x0
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alphas_cumprod_t = self._extract(
            self.sqrt_alphas_cumprod, timesteps, x0.shape
        )
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, timesteps, x0.shape
        )
        return sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise

    def step(self, x_t, timesteps, eps_pred, generator=None):
        """
        Compute one reverse diffusion step to obtain x_{t-1} from x_t.
        """
        sqrt_recip_alpha_t = self._extract(self.sqrt_recip_alphas, timesteps, x_t.shape)
        beta_t = self._extract(self.betas, timesteps, x_t.shape)
        sqrt_one_minus_alpha_bar_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, timesteps, x_t.shape
        )
        posterior_variance_t = self._extract(
            self.posterior_variance, timesteps, x_t.shape
        )

        # Mean as in DDPM Eq. 11
        model_mean = sqrt_recip_alpha_t * (
            x_t - (beta_t / (sqrt_one_minus_alpha_bar_t + 1e-8)) * eps_pred
        )

        # For t > 0, add noise; for t == 0, no noise
        if generator is None:
            noise = torch.randn_like(x_t)
        else:
            noise = torch.randn(x_t.shape, device=x_t.device, generator=generator)

        nonzero_mask = (timesteps != 0).float().view(-1, *([1] * (x_t.ndim - 1)))
        x_prev = model_mean + nonzero_mask * torch.sqrt(posterior_variance_t) * noise
        return x_prev

    @staticmethod
    def _extract(a, t, x_shape):
        """Extract coefficients a_t for batch of indices t, reshape to x_shape."""
        out = a.gather(-1, t)
        return out.view(-1, *([1] * (len(x_shape) - 1)))

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
