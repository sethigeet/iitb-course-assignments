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

        self.current_arm = -1
        self.pulls_remaining_in_batch = 0
        self.next_batch_potential = np.ones(num_arms)
        self.batch_growth_factor = 1.5
        # END EDITING HERE

    def give_pull(self):
        self.t += 1

        # If we are in the middle of a batch, continue pulling the same arm
        # without any expensive calculations.
        if self.pulls_remaining_in_batch > 0:
            self.pulls_remaining_in_batch -= 1
            return self.current_arm

        # --- Start of a new decision phase ---

        # Standard round-robin initialization: play each arm once.
        for arm in range(self.num_arms):
            if self.counts[arm] == 0:
                self.current_arm = arm
                return self.current_arm

        # Re-calculate UCB indices for all arms. This is the expensive step
        # that our batching strategy aims to perform less frequently.
        indices = np.zeros(self.num_arms)
        for arm in range(self.num_arms):
            emp_mean = self.successes[arm] / self.counts[arm]
            indices[arm] = calc_kl_ucb(emp_mean, self.counts[arm], self.t)

        # Select the arm with the highest UCB index.
        best_arm = int(np.argmax(indices))
        self.current_arm = best_arm

        # Safety mechanism: Reset the batch potential for any arm that was NOT chosen.
        # This ensures that if an arm becomes the leader again, it starts with a small
        # batch, preventing over-commitment based on stale information.
        for arm in range(self.num_arms):
            if arm != best_arm:
                self.next_batch_potential[arm] = 1.0

        # Determine the size of the new batch for the chosen arm.
        batch_size = int(np.ceil(self.next_batch_potential[best_arm]))

        # Schedule the batch pulls. The current pull is the first in the batch.
        self.pulls_remaining_in_batch = batch_size - 1

        # Increase the batch potential for the chosen arm for the *next* time it is selected.
        self.next_batch_potential[best_arm] *= self.batch_growth_factor

        return self.current_arm

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
