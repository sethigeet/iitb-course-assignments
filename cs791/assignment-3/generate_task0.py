"""
Task 0: Core Baseline Decoding Implementations

Implemented Algorithms:
    1. Greedy Decoding: Deterministic selection of maximum likelihood tokens
    2. Temperature Sampling: Stochastic sampling with temperature-scaled distributions
    3. Top-k Sampling: Restricted stochastic sampling from k most probable tokens
"""

import torch

from utils import load_model as utils_load_model

# Re-export load_model from utils
load_model = utils_load_model


@torch.no_grad()
def greedy_decode(tokenizer, model, prefix: str, max_new: int, eos_id: int) -> str:
    """Perform greedy decoding for deterministic text generation.

    Args:
        tokenizer (AutoTokenizer): HuggingFace tokenizer for encoding/decoding
        model (AutoModelForCausalLM): Causal language model in evaluation mode
        prefix (str): Input text prompt to continue
        max_new (int): Maximum number of new tokens to generate
        eos_id (int): End-of-sequence token ID for early termination

    Returns:
        str: Generated text continuation (excluding input prefix)
    """
    input_ids = tokenizer.encode(prefix, return_tensors="pt").to(model.device)
    input_ids_size = len(input_ids[0])

    for _ in range(max_new):
        outputs = model(input_ids)
        logits = outputs.logits[0, -1, :]

        # Select the token with highest probability (greedy)
        next_token_id = torch.argmax(logits).item()

        # Stop if EOS token is reached
        if next_token_id == eos_id:
            break

        # Update input_ids for next iteration
        input_ids = torch.cat(
            [input_ids, torch.tensor([[next_token_id]]).to(model.device)], dim=1
        )

    return tokenizer.decode(input_ids[0][input_ids_size:], skip_special_tokens=True)


@torch.no_grad()
def temperature_decode(
    tokenizer, model, prefix: str, max_new: int, eos_id: int, tau: float
) -> str:
    """Perform temperature sampling for stochastic text generation.

    Args:
        tokenizer (AutoTokenizer): HuggingFace tokenizer for encoding/decoding
        model (AutoModelForCausalLM): Causal language model in evaluation mode
        prefix (str): Input text prompt to continue
        max_new (int): Maximum number of new tokens to generate
        eos_id (int): End-of-sequence token ID for early termination
        tau (float): Temperature parameter for scaling logits (must be > 0)

    Returns:
        str: Generated text continuation (excluding input prefix)
    """
    input_ids = tokenizer.encode(prefix, return_tensors="pt").to(model.device)
    input_ids_size = len(input_ids[0])

    for _ in range(max_new):
        outputs = model(input_ids)
        logits = outputs.logits[0, -1, :]

        # Scale logits by temperature and convert to probabilities
        scaled_logits = logits / tau
        probs = torch.softmax(scaled_logits, dim=-1)

        # Sample from the probability distribution
        next_token_id = torch.multinomial(probs, 1).item()

        # Stop if EOS token is reached
        if next_token_id == eos_id:
            break

        # Update input_ids for next iteration
        input_ids = torch.cat(
            [input_ids, torch.tensor([[next_token_id]]).to(model.device)], dim=1
        )

    return tokenizer.decode(input_ids[0][input_ids_size:], skip_special_tokens=True)


@torch.no_grad()
def topk_decode(
    tokenizer, model, prefix: str, max_new: int, eos_id: int, k: int
) -> str:
    """Perform top-k sampling for controlled diversity in text generation.

    Args:
        tokenizer (AutoTokenizer): HuggingFace tokenizer for encoding/decoding
        model (AutoModelForCausalLM): Causal language model in evaluation mode
        prefix (str): Input text prompt to continue
        max_new (int): Maximum number of new tokens to generate
        eos_id (int): End-of-sequence token ID for early termination
        k (int): Number of top tokens to consider (must be >= 1)

    Returns:
        str: Generated text continuation (excluding input prefix)
    """
    input_ids = tokenizer.encode(prefix, return_tensors="pt").to(model.device)
    input_ids_size = len(input_ids[0])

    for _ in range(max_new):
        outputs = model(input_ids)
        logits = outputs.logits[0, -1, :]

        # Get top-k logits and indices and convert to probabilities
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

    return tokenizer.decode(input_ids[0][input_ids_size:], skip_special_tokens=True)
