"""Sequential Importance Sampling (Algorithm 1) — generation utilities."""

from __future__ import annotations

import os
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    raise NotImplementedError("Students must implement this function.")


@torch.no_grad()
def topk_decode_ids(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    prefix: str,
    max_new: int,
    k: int,
    eos_id: int,
) -> List[int]:
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
    raise NotImplementedError("Students must implement this function.")


def importance_sampling_for_prompt(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
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
    raise NotImplementedError("Students must implement this function.")
