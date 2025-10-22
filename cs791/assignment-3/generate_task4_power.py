"""Power Distribution Sampling with Metropolis-Hastings MCMC — generation utilities.

This module implements the power distribution sampling algorithm described in:
"Reasoning with Sampling: Your Base Model is Smarter Than You Think" (https://arxiv.org/html/2510.14901v1)

The algorithm samples from p^alpha(x) where p(x) is the base model distribution
and alpha is the power parameter that controls the sharpening of the distribution.
"""

import math
import random
from typing import Dict, List, Tuple

import torch

from utils import load_model as utils_load_model

# Re-export load_model from utils
load_model = utils_load_model


@torch.no_grad()
def compute_sequence_log_probability(
    tokenizer, model, prefix: str, continuation_ids: List[int]
) -> float:
    """Compute the log probability of a sequence under the base model.

    Args:
        tokenizer: HuggingFace tokenizer
        model: Language model
        prefix: Input prompt
        continuation_ids: Token IDs of the continuation

    Returns:
        Log probability of the sequence
    """
    # Encode the full sequence
    full_text = prefix + tokenizer.decode(continuation_ids, skip_special_tokens=True)
    input_ids = tokenizer.encode(full_text, return_tensors="pt").to(model.device)

    # Get log probabilities for each token
    outputs = model(input_ids)
    logits = outputs.logits[0, :-1, :]  # Exclude last token (no target for it)
    targets = input_ids[0, 1:]  # Shift by 1 for next token prediction

    # Compute log probabilities
    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

    # Sum log probabilities for the continuation part
    prefix_length = len(tokenizer.encode(prefix, return_tensors="pt")[0])
    continuation_log_probs = token_log_probs[prefix_length:]

    return continuation_log_probs.sum().item()


@torch.no_grad()
def sample_continuation_with_topk(
    tokenizer, model, prefix: str, max_new: int, k: int, eos_id: int
) -> List[int]:
    """Sample a continuation using top-k sampling.

    Args:
        tokenizer: HuggingFace tokenizer
        model: Language model
        prefix: Input prompt
        max_new: Maximum number of new tokens
        k: Top-k parameter
        eos_id: End-of-sequence token ID

    Returns:
        List of token IDs for the continuation
    """
    input_ids = tokenizer.encode(prefix, return_tensors="pt").to(model.device)
    input_ids_size = len(input_ids[0])

    for _ in range(max_new):
        outputs = model(input_ids)
        logits = outputs.logits[0, -1, :]

        # Get top-k logits and indices
        top_k_logits, top_k_indices = torch.topk(logits, k)
        probs = torch.softmax(top_k_logits, dim=-1)

        # Sample from the top-k distribution
        sampled_idx = torch.multinomial(probs, 1)
        next_token_id = top_k_indices[sampled_idx[0]].item()

        # Stop if EOS token is reached
        if next_token_id == eos_id:
            break

        # Update input_ids for next iteration
        input_ids = torch.cat(
            [input_ids, torch.tensor([[next_token_id]]).to(model.device)], dim=1
        )

    return input_ids[0, input_ids_size:].tolist()


@torch.no_grad()
def metropolis_hastings_step(
    tokenizer,
    model,
    prefix: str,
    current_ids: List[int],
    max_new: int,
    alpha: float,
    k: int,
    eos_id: int,
) -> Tuple[List[int], bool]:
    """Perform one Metropolis-Hastings step for power distribution sampling.

    Args:
        tokenizer: HuggingFace tokenizer
        model: Language model
        prefix: Input prompt
        current_ids: Current sequence token IDs
        max_new: Maximum number of new tokens
        alpha: Power parameter for p^alpha(x)
        k: Top-k parameter for proposal
        eos_id: End-of-sequence token ID

    Returns:
        Tuple of (new_ids, accepted) where accepted indicates if the proposal was accepted
    """
    # Generate proposal using top-k sampling
    proposal_ids = sample_continuation_with_topk(
        tokenizer, model, prefix, max_new, k, eos_id
    )

    # Compute log probabilities under the base model
    current_log_prob = compute_sequence_log_probability(
        tokenizer, model, prefix, current_ids
    )
    proposal_log_prob = compute_sequence_log_probability(
        tokenizer, model, prefix, proposal_ids
    )

    # Compute acceptance probability for power distribution p^alpha(x)
    # The acceptance ratio is: min(1, (p_proposal^alpha / p_current^alpha) * (q_current|proposal / q_proposal|current))
    # For symmetric proposals (top-k sampling), the proposal ratio cancels out
    log_acceptance_ratio = alpha * (proposal_log_prob - current_log_prob)

    # Accept or reject the proposal
    if log_acceptance_ratio >= 0:
        accepted = True
    else:
        acceptance_prob = math.exp(log_acceptance_ratio)
        accepted = random.random() < acceptance_prob

    if accepted:
        return proposal_ids, True
    else:
        return current_ids, False


def power_mcmc_for_prompt(
    tokenizer,
    model,
    *,
    prefix: str,
    B: int,
    max_new_tokens: int,
    eos_id: int,
    alpha: float,
    steps: int,
    k: int,
) -> Dict:
    """Run power distribution MCMC sampling for one prompt.

    Args:
        tokenizer: HuggingFace tokenizer
        model: Language model
        prefix: Input prompt
        B: Number of samples to generate
        max_new_tokens: Maximum number of new tokens
        eos_id: End-of-sequence token ID
        alpha: Power parameter for p^alpha(x)
        steps: Number of MCMC steps
        k: Top-k parameter for proposal

    Returns:
        Dict containing samples and normalized weights
    """
    samples = []
    weights = []

    for _ in range(B):
        # Generate the initial output from the model using top-k sampling
        current_ids = sample_continuation_with_topk(
            tokenizer, model, prefix, max_new_tokens, k, eos_id
        )

        # Run MCMC steps
        for _ in range(steps):
            current_ids, accepted = metropolis_hastings_step(
                tokenizer, model, prefix, current_ids, max_new_tokens, alpha, k, eos_id
            )

        # Compute final weight (proportional to p^alpha(x))
        log_prob = compute_sequence_log_probability(
            tokenizer, model, prefix, current_ids
        )
        weight = math.exp(alpha * log_prob)

        samples.append(
            {
                "text": tokenizer.decode(current_ids, skip_special_tokens=True),
                "weight": weight,
            }
        )
        weights.append(weight)

    # Normalize weights
    total_weight = sum(weights)
    normalized_weights = (
        [w / total_weight for w in weights] if total_weight > 0 else [1.0 / B] * B
    )

    return {
        "samples": samples,
        "normalized_weights": normalized_weights,
    }
