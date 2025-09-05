from typing import Dict, List, Optional, Tuple

import numpy as np

# =========================================================
# ===============   ENVIRONMENT (Poisson)   ===============
# =========================================================


class PoissonDoorsEnv:
    """
    This creates a Poisson environment. There are K doors and each has an associated mean.
    In each step you pick an arm i. Damage to a door is drawn from its corresponding
    Poisson Distribution. Initial health of each door is H0 and decreases by damage in each step.
    Game ends when any door's health < 0.
    """

    def __init__(
        self, mus: List[float], H0: int = 100, rng: Optional[np.random.Generator] = None
    ):
        self.mus = np.array(mus, dtype=float)
        assert np.all(self.mus > 0), "Poisson means must be > 0"
        self.K = len(mus)
        self.H0 = H0
        self.rng = rng if rng is not None else np.random.default_rng()
        self.reset()

    def reset(self):
        self.health = np.full(self.K, self.H0, dtype=float)
        self.t = 0
        return self.health.copy()

    def step(self, arm: int) -> Tuple[float, bool, Dict]:
        reward = float(self.rng.poisson(self.mus[arm]))
        self.health[arm] -= reward
        self.t += 1
        done = bool(np.any(self.health < 0.0))
        return (
            reward,
            done,
            {"reward": reward, "health": self.health.copy(), "t": self.t},
        )


# =========================================================
# =====================   POLICIES   ======================
# =========================================================


class Policy:
    """
    Base Policy interface.
    - Implement select_arm(self, t) to return an int in [0, K-1] to choose an arm.
    - Optionally override update(...) for custom learning.
    """

    def __init__(self, K: int, rng: Optional[np.random.Generator] = None):
        self.K = K
        self.rng = rng if rng is not None else np.random.default_rng()
        self.counts = np.zeros(K, dtype=int)
        self.sums = np.zeros(K, dtype=float)

    def reset_stats(self):
        self.counts[:] = 0
        self.sums[:] = 0.0

    def update(self, arm: int, reward: float):
        self.counts[arm] += 1
        self.sums[arm] += reward

    @property
    def means(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.sums / np.maximum(self.counts, 1)

    def select_arm(self, t: int) -> int:
        raise NotImplementedError


## TASK 2: Make changes here to implement your policy ###
class StudentPolicy(Policy):
    """
    Implement your own algorithm here.
    Replace select_arm with your strategy.
    Currently it has a simple implementation of the epsilon greedy strategy.
    Change this to implement your algorithm for the problem.
    """

    def __init__(self, K: int, rng: Optional[np.random.Generator] = None):
        super().__init__(K, rng)
        # Gamma prior for Poisson mean (conjugate): mu ~ Gamma(alpha0, beta0)
        self.alpha0 = 1.0
        self.beta0 = 1.0
        self.alpha = np.full(K, self.alpha0, dtype=float)
        self.beta = np.full(K, self.beta0, dtype=float)
        self.health = None  # will be populated from env via update(..., health)
        self._refined_done = False

    def reset_stats(self):
        super().reset_stats()
        self.alpha[:] = self.alpha0
        self.beta[:] = self.beta0
        self.health = None
        self._refined_done = False

    def select_arm(self, t: int) -> int:
        # Ensure each arm is tried at least once (minimal exploration)
        for a in range(self.K):
            if self.counts[a] == 0:
                return a

        # Use current health from env if available; otherwise assume equal healths
        if self.health is None:
            health = np.ones(self.K, dtype=float)
        else:
            health = self.health.astype(float)

        # Posterior mean of Poisson rate under Gamma(alpha, beta)
        beta_safe = np.maximum(self.beta, 1e-9)
        mu_hat = self.alpha / beta_safe

        # Minimal refinement: after each arm has 1 sample, give top-2 arms one extra sample if needed
        if not self._refined_done:
            top2 = np.argsort(-mu_hat)[: min(2, self.K)]
            for a in top2:
                if self.counts[a] < 2:
                    return int(a)
            self._refined_done = True

        # Commit: choose arm minimizing estimated remaining steps health_i / mu_hat_i
        denom = np.maximum(mu_hat, 1e-9)
        remaining_steps_est = health / denom
        return int(np.argmin(remaining_steps_est))

    def update(self, arm: int, reward: float, health: Optional[np.ndarray] = None):
        # Update generic stats
        super().update(arm, reward)

        # Update Gamma posterior for the struck arm
        self.alpha[arm] += float(reward)
        self.beta[arm] += 1.0

        # Keep track of current health from the environment (if provided)
        if health is not None:
            self.health = np.array(health, dtype=float)
