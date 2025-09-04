"""
Task 3: Optimized KL-UCB Implementation

This file implements both standard and optimized KL-UCB algorithms for multi-armed bandits.
The optimized version aims to reduce computational overhead while maintaining good regret performance.
"""

import math
import numpy as np
import matplotlib.pyplot as plt

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

# ------------------ Optimized KL-UCB Algorithm ------------------

class KL_UCB_Optimized(Algorithm):
    """
    Optimized KL-UCB algorithm that reduces computation while maintaining identical regret.
    This implements a batched KL-UCB with exponential+binary search for safe pulls of the current best arm.
    """
    ## You can define other functions also in the class if needed
    
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # can initialize member variables here
        #START EDITING HERE
        #END EDITING HERE
    
    def give_pull(self):
        #START EDITING HERE
        pass
        #END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        #START EDITING HERE
        pass
        #END EDITING HERE

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
        #START EDITING HERE
        #END EDITING HERE
    
    def give_pull(self):
        #START EDITING HERE
        pass
        #END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        #START EDITING HERE
        pass
        #END EDITING HERE
