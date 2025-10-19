#!/usr/bin/env python3

import math
import pickle
from typing import Dict, List


class FastRewardCalculator:
    def __init__(self, cache_file: str, epsilon: float = 1e-9):
        """
        cache_file: pickle with at least
          - 'trigram_probs': Dict[str, float], key = "tok1,tok2,tok3", value = P(t3|t1,t2)
        """
        with open(cache_file, "rb") as f:
            cache = pickle.load(f)
        self._tri_probs: Dict[str, float] = cache["trigram_probs"]
        self._eps: float = float(epsilon)

        # Expose a token LM object with .logp expected by SMC/TSMC code.
        # Keep naming stable: reward_calc.token_lm.logp(...)
        self.token_lm = _TokenLM(self._tri_probs, self._eps)

    def calculate_reward_tokens(
        self, tokens: List[str], normalize: bool = True
    ) -> float:
        """
        Args:
            tokens (List[str]):
                List of token strings

            normalize (bool, optional):
                Whether to compute the average reward per trigram (True)
                or the unnormalized total reward (False).

        Returns:
            float:
                Returns 0.0 if fewer than 3 tokens are provided.
        """
        if len(tokens) < 3:
            return 0.0

        reward = 0.0
        for i in range(len(tokens) - 2):
            reward += self.token_lm.logp(tokens[i], tokens[i + 1], tokens[i + 2])

        if normalize:
            reward = reward / len(tokens)

        return reward


class _TokenLM:
    """Minimal token-trigram LM with logp only. Internal use."""

    def __init__(self, tri_probs: Dict[str, float], eps: float):
        self._tri = tri_probs
        self._eps = eps

    @staticmethod
    def _key(t1: str, t2: str, t3: str) -> str:
        return f"Ġ{t1},Ġ{t2},Ġ{t3}"

    def logp(self, t1: str, t2: str, t3: str) -> float:
        """Return log P(t3 | t1, t2) with epsilon floor."""
        p = self._tri.get(self._key(t1, t2, t3), 0.0)
        if p <= 0.0:
            return self._eps
        return -math.log(p)
