"""Twisted Sequential Monte Carlo (TSMC) implementation - simplified for assignment."""

from __future__ import annotations

from typing import Any, Dict, List

import torch

from api import FastRewardCalculator
from utils import load_counts_and_reward as utils_load_counts_and_reward
from utils import load_model as utils_load_model

# Re-export load_model and load_counts_and_reward from utils
load_model = utils_load_model
load_counts_and_reward = utils_load_counts_and_reward


def cal_intermediate_target_dist(
    reward_calc: FastRewardCalculator, tokenizer, full_ids: List[int]
) -> float:
    """
    Args:
        reward_calc: FastRewardCalculator (token_lm.logp available).
        tokenizer: for ids→tokens conversion.
        full_ids: current full context ids (prompt + generated so far).

    Returns:
        float ΔR_t ≥ 0.
    """
    raise NotImplementedError("Students must implement this function.")


def cal_twist_function(
    reward_calc: FastRewardCalculator, tokenizer, seq_ids: List[int]
) -> float:
    """
    Inputs:
        reward_calc: FastRewardCalculator with token_lm access.
        tokenizer: to convert ids→tokens.
        seq_ids: current full context ids (prompt + generated).

    Returns:
        Expected positive delta (float) ≥ 0. For t < 2, you may return 0.0.

    Note:
        you are allowed to define additional helper functions if needed in FastRewardCalculator class for calculation of expectation.
    """
    raise NotImplementedError("Students must implement this function.")


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
    raise NotImplementedError("Students must implement this function.")
