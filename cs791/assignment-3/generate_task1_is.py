"""Sequential Importance Sampling (Algorithm 1) — generation utilities."""

from __future__ import annotations

import math
import os
from typing import Dict, List

import torch

# Import the fast trigram API
from api import FastRewardCalculator
from utils import load_model as utils_load_model

# Re-export load_model from utils
load_model = utils_load_model


def load_counts_and_reward(
    counts_dir: str, epsilon: float = 1e-9
) -> FastRewardCalculator:
    """Initialize trigram-based reward calculator for Sequential Importance Sampling.

    Args:
        counts_dir: Directory path containing ngrams data with trigram_probs.pkl cache
        epsilon: Smoothing parameter - minimum probability for unseen trigrams (prevents log(0))

    Returns:
        FastRewardCalculator: Configured calculator for computing R(x) rewards
    """
    cache_file = os.path.join(counts_dir, "trigram_probs.pkl")
    return FastRewardCalculator(cache_file, epsilon=epsilon)


def reward_sum_pos_ids(
    reward_calc: FastRewardCalculator, tokenizer, ids: List[int]
) -> float:
    """Compute positive reward on token ids: R_sum over token trigrams.

    Inputs:
        reward_calc: FastRewardCalculator (token_lm.logp available).
        tokenizer: used only to convert ids→tokens.
        ids: full scored context (prompt+continuation) token ids.

    Output:
        R_sum (float). If len(ids) < 3, return 0.0.
    """
    if len(ids) < 3:
        return 0.0

    tokens = tokenizer.decode(ids, skip_special_tokens=True)
    reward = reward_calc.calculate_reward_tokens(
        tokens.strip().split(" "), normalize=True
    )
    return reward


@torch.no_grad()
def batched_topk_decode_ids(
    tokenizer,
    model,
    prefix: str,
    max_new: int,
    k: int,
    batch_size: int,
    eos_id: int,
) -> List[List[int]]:
    """Sample one continuation with top-k proposal. Return continuation token ids (EOS excluded).

    Inputs:
        tokenizer, model: HF components from load_model.
        prefix: full prompt string fed to the model.
        max_new: continuation token budget.
        k: top-k size.
        eos_id: stopping id. Stop early if sampled.

    Output:
        gen_ids: List[int] of sampled token ids for the continuation.
    """
    input_ids = tokenizer.encode(prefix, return_tensors="pt").to(model.device)
    input_tokens_size = input_ids.shape[1]
    input_ids = input_ids.repeat(batch_size, 1)

    completed_ids = torch.zeros(batch_size).to(model.device)

    for _ in range(max_new):
        outputs = model(input_ids)
        logits = outputs.logits[:, -1, :]

        # Get top-k logits and indices and convert to probabilities
        top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)
        probs = torch.softmax(top_k_logits, dim=-1)

        # Sample from the top-k distribution
        next_token_ids = []
        for i in range(batch_size):
            if completed_ids[i]:
                next_token_ids.append(model.config.eos_token_id)
                continue

            sampled_idx = torch.multinomial(probs[i], 1)
            next_token_id = top_k_indices[i, sampled_idx]
            next_token_ids.append(next_token_id)

        next_token_ids = torch.tensor(next_token_ids).to(model.device)
        input_ids = torch.cat(
            [
                input_ids,
                next_token_ids.unsqueeze(-1),
            ],
            dim=-1,
        )

        # Stop if all sequences are completed
        completed_ids = completed_ids.masked_fill(
            next_token_ids == model.config.eos_token_id, 1
        )
        if completed_ids.all():
            break

    return input_ids[:, input_tokens_size:].tolist()


def importance_sampling_for_prompt(
    tokenizer,
    model,
    reward_calc: FastRewardCalculator,
    *,
    prefix: str,
    K: int,
    max_new_tokens: int,
    eos_id: int,
    beta: float,
    k: int,
) -> Dict:
    """Run SIS for one prompt and return samples with weights.

    Inputs:
        tokenizer, model, reward_calc: initialized components.
        prefix: full prompt string given to the model (instruction + space + prefix).
        K: number of continuations to sample.
        max_new_tokens: continuation budget.
        eos_id: end-of-sequence id.
        beta: reward scale.
        k: top-k for proposal.

    Output dict:
        {
            "samples": [
                {"text": str, "weight": float},
                ...
            ],
            "normalized_weights": [float, ...]   # length K
        }
    """

    samples = []
    weights = []
    gen_ids = batched_topk_decode_ids(
        tokenizer, model, prefix, max_new_tokens, k, K, eos_id
    )
    for i in range(K):
        reward = reward_sum_pos_ids(reward_calc, tokenizer, gen_ids[i])
        weight = math.exp(beta * reward)
        samples.append(
            {
                "text": tokenizer.decode(gen_ids[i], skip_special_tokens=True),
                "weight": weight,
            }
        )
        weights.append(weight)

    total_weights = sum(weights)
    normalized_weights = list([w / total_weights for w in weights])

    return {
        "samples": samples,
        "normalized_weights": normalized_weights,
    }
