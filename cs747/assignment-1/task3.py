"""
Task 3: Optimized KL-UCB Implementation

This file implements both standard and optimized KL-UCB algorithms for multi-armed bandits.
The optimized version aims to reduce computational overhead while maintaining good regret performance.
"""

import math

import numpy as np

# ------------------ Base Algorithm Class ------------------


class Algorithm:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        self.horizon = horizon

    def give_pull(self):
        raise NotImplementedError

    def get_reward(self, arm_index, reward):
        raise NotImplementedError


# ------------------ KL-UCB utilities ------------------
## You can define other helper functions here if needed


def kl_div(p, q, eps=1e-12):
    """KL divergence between Bernoulli(p) and Bernoulli(q)."""
    p = min(max(p, eps), 1 - eps)
    q = min(max(q, eps), 1 - eps)
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


def calc_kl_ucb(empirical_mean, count, t, c=2.0, eps=1e-6, max_iter=25):
    target_kl = (
        math.log(t) + c * math.log(max(math.log(max(t, 2)), 1.0000001))
    ) / count

    # Binary search for the upper bound
    low = empirical_mean
    high = 1.0
    i = 0
    while i < max_iter and high - low > eps:
        mid = (low + high) / 2
        kl_val = kl_div(empirical_mean, mid)
        if abs(kl_val - target_kl) < eps:
            return mid
        elif kl_val < target_kl:
            low = mid
        else:
            high = mid

        i += 1

    return (low + high) / 2


# ------------------ Optimized KL-UCB Algorithm ------------------


class KL_UCB_Optimized(Algorithm):
    """
    Optimized KL-UCB algorithm that reduces computation while maintaining identical regret.
    This implements a batched KL-UCB with exponential+binary search for safe pulls of the current best arm.
    """

    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # START EDITING HERE
        self.counts = np.zeros(num_arms, dtype=int)
        self.successes = np.zeros(num_arms, dtype=float)
        self.t = 0

        self.kl_ucbs = None
        self.current_best_arm = -1
        self.pulls_remaining_in_batch = 0
        self.batch_size = 1.0
        self.batch_growth_factor = 1.5
        # END EDITING HERE

    def give_pull(self):
        self.t += 1

        # If we are in the middle of a batch, continue pulling the same arm
        # without any expensive calculations.
        if self.pulls_remaining_in_batch > 0:
            self.pulls_remaining_in_batch -= 1
            return self.current_best_arm

        # Pull each arm once
        for arm in range(self.num_arms):
            if self.counts[arm] == 0:
                self.current_best_arm = arm
                return self.current_best_arm

        if self.kl_ucbs is None:
            self.kl_ucbs = np.zeros(self.num_arms)
            for arm in range(self.num_arms):
                emp_mean = self.successes[arm] / self.counts[arm]
                self.kl_ucbs[arm] = calc_kl_ucb(emp_mean, self.counts[arm], self.t)

            self.current_best_arm = int(np.argmax(self.kl_ucbs))
            self.next_best_arm = int(
                np.argmax(
                    self.kl_ucbs[~(np.arange(self.num_arms) == self.current_best_arm)]
                )
            )

        # Re-calculate UCB indices for the next best arm
        self.kl_ucbs[self.current_best_arm] = calc_kl_ucb(
            self.successes[self.current_best_arm] / self.counts[self.current_best_arm],
            self.counts[self.current_best_arm],
            self.t,
        )
        self.kl_ucbs[self.next_best_arm] = calc_kl_ucb(
            self.successes[self.next_best_arm] / self.counts[self.next_best_arm],
            self.counts[self.next_best_arm],
            self.t,
        )

        # Select the arm with the highest UCB index
        best_arm = int(np.argmax(self.kl_ucbs))
        self.current_best_arm = best_arm

        # Reset the batch multiplier
        if self.current_best_arm != self.next_best_arm:
            self.batch_size = 1.0
            self.pulls_remaining_in_batch = 0
            self.next_best_arm = int(
                np.argmax(
                    self.kl_ucbs[~(np.arange(self.num_arms) == self.current_best_arm)]
                )
            )

        batch_size = int(np.ceil(self.batch_size))

        self.pulls_remaining_in_batch = batch_size - 1
        self.batch_size *= self.batch_growth_factor

        return self.current_best_arm

    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.counts[arm_index] += 1
        self.successes[arm_index] += reward
        # END EDITING HERE


# ------------------ Bonus KL-UCB Algorithm (Optional - 1 bonus mark) ------------------


class KL_UCB_Bonus(Algorithm):
    """
    BONUS ALGORITHM (Optional - 1 bonus mark)

    This algorithm must produce EXACTLY IDENTICAL regret trajectories to KL_UCB_Standard
    while achieving significant speedup. Students implementing this will earn 1 bonus mark.

    Requirements for bonus:
    - Must produce identical regret trajectories (checked with strict tolerance)
    - Must achieve specified speedup thresholds on bonus testcases
    - Must include detailed explanation in report
    """

    # You can define other functions also in the class if needed

    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # can initialize member variables here
        # START EDITING HERE
        # END EDITING HERE

    def give_pull(self):
        # START EDITING HERE
        pass
        # END EDITING HERE

    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        pass
        # END EDITING HERE
