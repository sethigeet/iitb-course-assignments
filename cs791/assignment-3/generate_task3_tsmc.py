"""Twisted Sequential Monte Carlo (TSMC) implementation - simplified for assignment."""

import math
from typing import Any, Dict

import torch

from api import FastRewardCalculator
from utils import load_counts_and_reward as utils_load_counts_and_reward
from utils import load_model as utils_load_model

# Re-export load_model and load_counts_and_reward from utils
load_model = utils_load_model
load_counts_and_reward = utils_load_counts_and_reward


def get_total_reward(reward_calc: FastRewardCalculator, tokenizer, ids) -> float:
    """
    Args:
        reward_calc: FastRewardCalculator (token_lm.logp available).
        tokenizer: for ids→tokens conversion.
        ids: current full context ids (prompt + generated so far).

    Returns:
        float ΔR_t ≥ 0.
    """
    if len(ids) < 3:
        return 0.0

    tokens = tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=True)
    reward = reward_calc.calculate_reward_tokens(tokens, normalize=True)
    return reward


def get_twisted_reward(reward_calc: FastRewardCalculator, tokenizer, ids) -> float:
    """
    Inputs:
        reward_calc: FastRewardCalculator with token_lm access.
        tokenizer: to convert ids→tokens.
        ids: current full context ids (prompt + generated).

    Returns:
        float: Expected positive delta (float) ≥ 0.
    """
    if len(ids) < 1:
        return 1.0

    tokens = tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=True)
    if len(tokens) == 1:
        reward = reward_calc.get_expected_reward(tokens[0])
    else:
        reward = reward_calc.get_expected_reward(tokens[-2], tokens[-1])

    return reward


@torch.no_grad()
def tsmc_for_prompt(
    tokenizer: Any,
    model: Any,
    reward_calc: Any,
    *,
    prefix: str,
    N: int,
    max_new_tokens: int,
    eos_id: int,
    beta: float,
    k: int,
) -> Dict:
    """Run TSMC for a single prompt.

    Inputs:
        tokenizer, model: HF components from load_model.
        reward_calc: FastRewardCalculator.
        prefix: full prompt string fed to the model (instruction + space + prefix).
        N: number of particles.
        max_new_tokens: continuation budget.
        eos_id: stopping id.
        beta: reward scale.
        k: top-k for base proposal.

    Output dict (minimal for eval):
        {
            "samples": [ {"text": str, "weight": float}, ... ],   # length N
            "normalized_weights": [float, ...]                    # softmax over final log_w
        }
    """

    input_ids = tokenizer.encode(prefix, return_tensors="pt").to(model.device)
    input_tokens_size = input_ids.shape[1]
    input_ids = input_ids.repeat(N, 1)

    completed_ids = torch.zeros(N).to(model.device)

    # Define here to have access to the weights variable outside the loop
    weights = []
    normalized_weights = torch.tensor([])
    total_probabilities = torch.ones(N).to(model.device)

    for t in range(max_new_tokens):
        outputs = model(input_ids)
        logits = outputs.logits[:, -1, :]

        # Get top-k logits and indices and convert to probabilities
        top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)
        probs = torch.softmax(top_k_logits, dim=-1)

        # Sample from the top-k distribution
        next_token_ids = []
        weights = []
        for i in range(N):
            if completed_ids[i]:
                next_token_ids.append(model.config.eos_token_id)
                weights.append(1.0)
                continue

            sampled_idx = torch.multinomial(probs[i], 1)

            next_token_id = top_k_indices[i, sampled_idx].item()
            next_token_prob = top_k_logits[i, sampled_idx].item()
            next_token_ids.append(next_token_id)
            total_probabilities[i] *= next_token_prob

            updated_input_ids = torch.cat(
                [
                    input_ids[i],
                    torch.tensor([next_token_id]).to(model.device),
                ]
            )
            if i < max_new_tokens - 1:
                pi_t = math.exp(
                    beta * get_twisted_reward(reward_calc, tokenizer, updated_input_ids)
                )
            else:
                pi_t = total_probabilities[i] * math.exp(
                    beta * get_total_reward(reward_calc, tokenizer, updated_input_ids)
                )

            pi_t_1 = math.exp(
                beta * get_twisted_reward(reward_calc, tokenizer, input_ids[i])
            )
            weight = pi_t / (pi_t_1 * next_token_prob)
            weights.append(weight)

        next_token_ids = torch.tensor(next_token_ids).to(model.device)
        input_ids = torch.cat(
            [
                input_ids,
                next_token_ids.unsqueeze(-1),
            ],
            dim=-1,
        )

        normalized_weights = torch.tensor(weights) / sum(weights)

        # Resample
        resampled_indices = torch.multinomial(
            normalized_weights, N, replacement=True
        ).to(input_ids.device)
        input_ids = input_ids[resampled_indices]
        total_probabilities = total_probabilities[resampled_indices]

        # Stop if all sequences are completed
        completed_ids = completed_ids.masked_fill(next_token_ids == eos_id, 1)
        if completed_ids.all():
            break

    gen_ids = input_ids[:, input_tokens_size:].tolist()
    normalized_weights = normalized_weights.tolist()

    samples = []
    for i in range(N):
        samples.append(
            {
                "text": tokenizer.decode(gen_ids[i], skip_special_tokens=True),
                "weight": weights[i],
            }
        )

    return {
        "samples": samples,
        "normalized_weights": normalized_weights,
    }
