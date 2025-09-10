from typing import Dict, List, Optional, Tuple, overload

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

    @overload
    def update(self, arm: int, reward: float) -> None: ...

    @overload
    def update(self, arm: int, reward: float, health: list[float]) -> None: ...

    def update(
        self, arm: int, reward: float, health: list[float] | None = None
    ) -> None:
        if health is not None:
            self.health = np.array(health, dtype=float)

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
        self.health = np.full(K, 100.0, dtype=float)

    def reset_stats(self):
        super().reset_stats()
        self.health = np.full(self.K, 100.0, dtype=float)

    def select_arm(self, t: int) -> int:
        # Make sure all arms have been tried at least once
        for a in range(self.K):
            if self.counts[a] == 0:
                return a

        # mu_hat = np.random.gamma(self.sums, self.counts)
        mu_hat = self.sums / self.counts
        mu_hat = np.maximum(mu_hat, 1e-9)  # Prevent division by zero
        # Return the door with the least expected remaining steps
        return int(np.argmin(self.health / np.sqrt(mu_hat)))

    def update(
        self, arm: int, reward: float, health: list[float] | None = None
    ) -> None:
        # Update generic stats
        super().update(arm, reward)

        if health is not None:
            self.health = np.array(health, dtype=float)
