import math
import time

import torch
from torch.nn import functional as F


def my_scaled_dot_product_attention(Q: torch.tensor, K: torch.tensor, V: torch.tensor):
    T, hs = Q.shape

    M = 32 * 1024  # on-chip SRAM budget in floats
    Bc = M // (4 * hs)
    Br = min(Bc, hs)
    Tc = (T + Bc - 1) // Bc  # number of K/V tiles
    Tr = (T + Br - 1) // Br  # number of Q tiles

    # m: row-wise max (initialized to negative infinity)
    # l: row-wise sum of exponentials (initialized to 0)
    # O: output accumulator (initialized to 0)
    m = torch.full((T, 1), -torch.inf, device=Q.device)
    l = torch.zeros((T, 1), device=Q.device)
    O = torch.zeros((T, hs), device=Q.device)

    scale = 1.0 / math.sqrt(hs)

    # Outer loop iterates over Tc tiles of K and V
    for j in range(Tc):
        # 1. Load Kj and Vj (Shape: Bc x D)
        start_j = j * Bc
        end_j = min(start_j + Bc, T)
        K_j = K[start_j:end_j, :]
        V_j = V[start_j:end_j, :]

        # Inner loop iterates over Tr tiles of Q
        for i in range(Tr):
            # 2. Load Qi, Oi, mi, li (Shape: Br x D, except m and l are Br x 1)
            start_i = i * Br
            end_i = min(start_i + Br, T)

            Q_i = Q[start_i:end_i, :]
            O_i = O[start_i:end_i, :]
            m_i = m[start_i:end_i, :]
            l_i = l[start_i:end_i, :]

            # 3. Compute tile scores (Shape: Br x Bc)
            S_ij = (Q_i @ K_j.T) * scale

            # 4. Compute tile statistics
            m_tilde_ij, _ = torch.max(S_ij, dim=-1, keepdim=True)  # Shape: Br x 1
            P_tilde_ij = torch.exp(S_ij - m_tilde_ij)  # Shape: Br x Bc
            l_tilde_ij = torch.sum(P_tilde_ij, dim=-1, keepdim=True)  # Shape: Br x 1

            # 5. Update running statistics and output using online softmax correction
            # Calculate the new running max
            m_new = torch.maximum(m_i, m_tilde_ij)

            # Calculate rescaling factors for the old and new terms
            exp_diff_old = torch.exp(m_i - m_new)
            exp_diff_new = torch.exp(m_tilde_ij - m_new)

            # Update the running sum
            l_new = exp_diff_old * l_i + exp_diff_new * l_tilde_ij

            # Update the output:
            # (Unnormalize old O_i, add new weighted V_j, and renormalize with l_new)
            O_new = (
                O_i * l_i * exp_diff_old + (P_tilde_ij @ V_j) * exp_diff_new
            ) / l_new

            # Write updated blocks back to "HBM"
            O[start_i:end_i, :] = O_new
            l[start_i:end_i, :] = l_new
            m[start_i:end_i, :] = m_new

    return O


def run_tests():
    hs = 64
    Ns = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Running on: {device.upper()}")
    print(f"{'N':<6} | {'Correct':<10} | {'Avg Time (ms)':<15}")
    print("-" * 35)

    for N in Ns:
        # Generate test data
        q = torch.randn(N, hs, device=device)
        k = torch.randn(N, hs, device=device)
        v = torch.randn(N, hs, device=device)

        # 1. Numerical Correctness Check
        with torch.no_grad():
            # is_causal=True in F.sdpa matches the torch.tril logic
            expected = F.scaled_dot_product_attention(
                q.view(1, 1, N, hs),
                k.view(1, 1, N, hs),
                v.view(1, 1, N, hs),
                is_causal=True,
            )
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
