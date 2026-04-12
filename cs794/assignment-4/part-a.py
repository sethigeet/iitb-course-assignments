import math
import time

import torch
from torch.nn import functional as F


def my_scaled_dot_product_attention(q: torch.tensor, k: torch.tensor, v: torch.tensor):
    (B, nh, T, hs) = q.size()

    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    mask = torch.tril(torch.ones(T, T), device=q.device).view(1, 1, T, T)
    att = att.masked_fill(mask == 0, float("-inf"))
    att = F.softmax(att, dim=-1)
    y = att @ v

    return y


def run_tests():
    B, nh, hs = 8, 16, 64
    Ns = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Running on: {device.upper()}")
    print(f"{'N':<6} | {'Correct':<10} | {'Avg Time (ms)':<15}")
    print("-" * 35)

    for N in Ns:
        # Generate test data
        q = torch.randn(B, nh, N, hs, device=device)
        k = torch.randn(B, nh, N, hs, device=device)
        v = torch.randn(B, nh, N, hs, device=device)

        # 1. Numerical Correctness Check
        with torch.no_grad():
            # is_causal=True in F.sdpa matches the torch.tril logic
            expected = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            actual = my_scaled_dot_product_attention(q, k, v)
            is_correct = torch.allclose(actual, expected, atol=1e-3, rtol=1e-3)

        # 2. Performance Benchmarking (Average over 10 iterations)
        durations = []
        # Warmup pass
        _ = my_scaled_dot_product_attention(q, k, v)

        for _ in range(10):
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = my_scaled_dot_product_attention(q, k, v)
            if device == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            durations.append(end - start)

        avg_ms = (sum(durations) / 10) * 1000
        print(f"{N:<6} | {str(is_correct):<10} | {avg_ms:<15.4f}")


if __name__ == "__main__":
    run_tests()
