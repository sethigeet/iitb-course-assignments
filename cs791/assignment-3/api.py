#!/usr/bin/env python3

import math
import pickle
from typing import Dict, List, Optional


class FastRewardCalculator:
    def __init__(
        self,
        trigram_probs_file: str,
        expected_rewards_file: str,
        epsilon: float = 1e-9,
    ):
        """
        Args:
            trigram_probs_file: pickle with at least
              - 'trigram_probs': Dict[str, float], key = "tok1,tok2,tok3", value = P(t3|t1,t2)
            bigram_probs_file: pickle with at least
              - 'bigram_probs': Dict[str, float], key = "tok1,tok2", value = P(t2|t1)
            expected_rewards_file: json with at least
              - 'expected_rewards': Dict[str, float], key = "tok1,tok2", value = expected reward of (tok1,tok2)
        """
        with open(trigram_probs_file, "rb") as f:
            self._tri_probs: Dict[str, float] = pickle.load(f)["trigram_probs"]
        with open(expected_rewards_file, "rb") as f:
            self._expected_rewards: Dict[str, float] = pickle.load(f)
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

    def get_expected_reward(self, token1: str, token2: Optional[str] = None) -> float:
        if token2 is None:
            return self._expected_rewards.get(token1, 0.0)
        else:
            return self._expected_rewards.get(f"{token1},{token2}", 0.0)


class _TokenLM:
    """Minimal token-trigram LM with logp only. Internal use."""

    def __init__(self, tri_probs: Dict[str, float], eps: float):
        self._tri = tri_probs
        self._eps = eps

    @staticmethod
    def _key(t1: str, t2: str, t3: str) -> str:
        return f"{t1},{t2},{t3}"

    def logp(self, t1: str, t2: str, t3: str) -> float:
        """Return log P(t3 | t1, t2) with epsilon floor."""
        p = self._tri.get(self._key(t1, t2, t3), 0.0)
        if p <= 0.0:
            return self._eps
        return -math.log(p)
