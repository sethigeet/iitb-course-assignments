"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the base Algorithm class that all algorithms should inherit
from. Here are the method details:
    - __init__(self, num_arms, horizon): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.

    - give_pull(self): This method is called when the algorithm needs to
        select an arm to pull. The method should return the index of the arm
        that it wants to pull (0-indexed).

    - get_reward(self, arm_index, reward): This method is called just after the
        give_pull method. The method should update the algorithm's internal
        state based on the arm that was pulled and the reward that was received.
        (The value of arm_index is the same as the one returned by give_pull.)

We have implemented the epsilon-greedy algorithm for you. You can use it as a
reference for implementing your own algorithms.
"""

import math

import numpy as np

# Hint: math.log is much faster than np.log for scalars


class Algorithm:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        self.horizon = horizon

    def give_pull(self):
        raise NotImplementedError

    def get_reward(self, arm_index, reward):
        raise NotImplementedError


# Example implementation of Epsilon Greedy algorithm
class Eps_Greedy(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # Extra member variables to keep track of the state
        self.eps = 0.1
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)

    def give_pull(self):
        if np.random.random() < self.eps:
            return np.random.randint(self.num_arms)
        else:
            return np.argmax(self.values)

    def get_reward(self, arm_index, reward):
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value


# START EDITING HERE
# You can use this space to define any helper functions that you need


def kl_div(p, q, eps=1e-12):
    # Clamp p and q to avoid log(0)
    p = min(max(p, eps), 1 - eps)
    q = min(max(q, eps), 1 - eps)
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


def calc_kl_ucb(empirical_mean, count, t, c=3.0, eps=1e-6, max_iter=50):
    target_kl = (math.log(t) + c * math.log(math.log(t))) / count

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


# END EDITING HERE


class UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # START EDITING HERE
        self.counts = np.zeros(num_arms)
        self.means = np.zeros(num_arms)
        self.total_pulls = 0
        # END EDITING HERE

    def give_pull(self):
        # START EDITING HERE
        # If any arm hasn't been pulled yet, pull it
        for arm in range(self.num_arms):
            if self.counts[arm] == 0:
                return arm

        # Calculate UCB values for all arms
        ucb_values = np.zeros(self.num_arms)
        for arm in range(self.num_arms):
            exploration_bonus = math.sqrt(
                2 * math.log(self.total_pulls) / self.counts[arm]
            )
            ucb_values[arm] = self.means[arm] + exploration_bonus

        return np.argmax(ucb_values)
        # END EDITING HERE

    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.total_pulls += 1
        self.counts[arm_index] += 1

        # Update the mean of the arm that was pulled
        n = self.counts[arm_index]
        new_mean = ((n - 1) / n) * self.means[arm_index] + (1 / n) * reward
        self.means[arm_index] = new_mean
        # END EDITING HERE


class KL_UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # START EDITING HERE
        self.counts = np.zeros(num_arms)
        self.means = np.zeros(num_arms)
        self.total_pulls = 0
        # END EDITING HERE

    def give_pull(self):
        # START EDITING HERE
        # If any arm hasn't been pulled yet, pull it
        for arm in range(self.num_arms):
            if self.counts[arm] == 0:
                return arm

        # Calculate KL-UCB values for all arms
        kl_ucb_values = np.zeros(self.num_arms)
        for arm in range(self.num_arms):
            kl_ucb_values[arm] = calc_kl_ucb(
                self.means[arm], self.counts[arm], self.total_pulls
            )

        return np.argmax(kl_ucb_values)
        # END EDITING HERE

    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.counts[arm_index] += 1
        self.total_pulls += 1

        # Update the mean of the arm that was pulled
        n = self.counts[arm_index]
        new_mean = ((n - 1) / n) * self.means[arm_index] + (1 / n) * reward
        self.means[arm_index] = new_mean
        # END EDITING HERE


class Thompson_Sampling(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # START EDITING HERE
        self.successes = np.zeros(num_arms)
        self.failures = np.zeros(num_arms)
        # END EDITING HERE

    def give_pull(self):
        # START EDITING HERE
        # Sample from Beta distribution for each arm
        sampled_values = np.zeros(self.num_arms)
        for arm in range(self.num_arms):
            sampled_values[arm] = np.random.beta(
                1 + self.successes[arm], 1 + self.failures[arm]
            )

        return np.argmax(sampled_values)
        # END EDITING HERE

    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        if reward == 1:
            self.successes[arm_index] += 1
        else:
            self.failures[arm_index] += 1
        # END EDITING HERE
