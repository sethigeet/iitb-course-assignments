import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed=42):
    """
    Makes everything deterministic for reproducible results.
    DO NOT MODIFY THIS FUNCTION.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SlidingWindowCausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, window_size, seed=42):
        """
        Initializes the Sliding Window Causal Self-Attention layer.

        DO NOT CHANGE ANYTHING IN THIS FUNCTION.

        Args:
            n_embd: Embedding dimension (total across all heads)
            n_head: Number of attention heads
            window_size: Maximum number of previous tokens each token can attend to
            seed: Random seed for deterministic weight initialization
        """
        super().__init__()
        assert n_embd % n_head == 0

        set_seed(seed)

        self.n_embd = n_embd  # Embedding dimension (C)
        self.n_head = n_head  # Number of heads (H)
        self.window_size = window_size  # Window size (W)
        self.head_dim = n_embd // n_head  # Dimension per head (D)

        # Single linear layer that projects input to Q, K, V concatenated
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)

    def forward(self, x):
        """
        Forward pass of the Sliding Window Causal Self-Attention layer.

        Args:
            x: Input tensor of shape (B, T, C) where
               B = batch size
               T = sequence length
               C = embedding dimension (same as n_embd)

        Returns:
            y: Output tensor of shape (B, T, C)
        """
        B, T, C = x.size()
        device = x.device

        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # TODO: Create a sliding window causal mask
        # The mask should enforce: (1) causality - no attending to future tokens
        #                          (2) window constraint - only attend to last W tokens
        # torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size)
        mask = torch.zeros(T, T)
        for i in range(T):
            mask[T-i-1, max(0, T-self.window_size-i):T-i] = 1
        
        mask = mask.view(1, 1, T, T) # Reshape the mask to make pytorch use broadcasting

        att = att.masked_fill(mask == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        return y


# TESTING CODE (YOU CAN USE THIS TO DEBUG YOUR IMPLEMENTATION)
if __name__ == "__main__":
    print("Testing your implementation...")

    set_seed(42)
    model = SlidingWindowCausalSelfAttention(
        n_embd=64, n_head=4, window_size=3, seed=42
    )
    model.eval()

    set_seed(42)
    x = torch.randn(2, 8, 64)

    with torch.no_grad():
        output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output mean: {output.mean().item():.6f}")
    print(f"Output std: {output.std().item():.6f}")
    print("\nIf you see this output, your code at least runs!")
    print("Use the provided grader script to verify correctness.")
