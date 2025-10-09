import argparse
from typing import List, Tuple

import numpy as np
from pulp import PULP_CBC_CMD, LpMinimize, LpProblem, LpStatus, LpVariable, lpSum, value


class MDP:
    # States: 0, 1, ..., num_states-1
    num_states: int
    # Actions: 0, 1, ..., num_actions-1
    num_actions: int
    gamma: float
    mdptype: str
    terminal_states: List[int]

    # [state, action, next_state] -> reward
    rewards: np.ndarray[Tuple[int, int, int], np.dtype[np.float64]]
    # [state, action, next_state] -> probability
    transitions: np.ndarray[Tuple[int, int, int], np.dtype[np.float64]]

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        gamma: float,
        mdptype: str,
        terminal_states: List[int],
        rewards: np.ndarray[Tuple[int, int, int], np.dtype[np.float64]],
        transitions: np.ndarray[Tuple[int, int, int], np.dtype[np.float64]],
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.gamma = gamma
        self.mdptype = mdptype
        self.terminal_states = terminal_states
        self.rewards = rewards
        self.transitions = transitions

    @staticmethod
    def from_file(path: str) -> "MDP":
        num_states = None
        num_actions = None
        terminal_states = []

        with open(path, "r") as f:
            lines = f.readlines()

        num_states = int(lines[0].split()[1])
        num_actions = int(lines[1].split()[1])
        terminal_states = [int(tok) for tok in lines[2].split(" ")[1:]]
        if len(terminal_states) == 1 and terminal_states[0] == -1:
            terminal_states = []

        transitions = np.zeros((num_states, num_actions, num_states), dtype=np.float64)
        rewards = np.zeros((num_states, num_actions, num_states), dtype=np.float64)
        for line in lines[3:-2]:
            tokens = line.split()
            state = int(tokens[1])
            action = int(tokens[2])
            next_state = int(tokens[3])
            reward = float(tokens[4])
            prob = float(tokens[5])
            transitions[state, action, next_state] = prob
            rewards[state, action, next_state] = reward

        mdptype = lines[-2].split()[1]
        gamma = float(lines[-1].split()[1])

        return MDP(
            num_states,
            num_actions,
            gamma,
            mdptype,
            terminal_states,
            rewards,
            transitions,
        )

    def action_value(
        self,
        state: int,
        action: int,
        values: np.ndarray[Tuple[int], np.dtype[np.float64]],
    ) -> float:
        return float(
            np.sum(
                self.transitions[state, action, :]
                * (self.rewards[state, action, :] + self.gamma * values)
            )
        )


class Policy:
    num_states: int
    state_to_action: List[int]

    def __init__(self, num_states: int, state_to_action: List[int]):
        self.num_states = num_states
        self.state_to_action = state_to_action

    @staticmethod
    def from_file(path: str):
        with open(path, "r") as f:
            lines = f.readlines()
        num_states = len(lines)
        state_to_action = [int(tok) for tok in lines]
        return Policy(num_states, state_to_action)


def solve_hpi(mdp: MDP, epsilon: float = 1e-6) -> Tuple[List[float], List[int]]:
    policy = [0 for _ in range(mdp.num_states)]

    while True:
        values = evaluate_policy(mdp, policy)
        policy_changed = False

        for state in range(mdp.num_states):
            if state in mdp.terminal_states:
                continue

            best_action = policy[state]
            best_action_value = mdp.action_value(state, best_action, values)
            for action in range(mdp.num_actions):
                action_value = mdp.action_value(state, action, values)
                if action_value > best_action_value + epsilon:
                    best_action_value = action_value
                    best_action = action

            if best_action != policy[state]:
                policy[state] = best_action
                policy_changed = True

        if not policy_changed:
            return values.tolist(), policy


def solve_lp(mdp: MDP) -> Tuple[List[float], List[int]]:
    prob = LpProblem("MDP", LpMinimize)
    value_vars = [LpVariable(f"V_{state}") for state in range(mdp.num_states)]
    prob += lpSum(value_vars), "Objective"

    for state in range(mdp.num_states):
        if state in mdp.terminal_states:
            prob += value_vars[state] == 0, f"Terminal state {state} constraint"
            continue

        for action in range(mdp.num_actions):
            rhs = 0.0
            for next_state in range(mdp.num_states):
                rhs += mdp.transitions[state, action, next_state] * (
                    mdp.rewards[state, action, next_state]
                    + mdp.gamma * value_vars[next_state]
                )
            prob += (
                value_vars[state] >= rhs,
                f"State {state} action {action} constraint",
            )

    solver = PULP_CBC_CMD(msg=False)
    status = prob.solve(solver)
    if LpStatus[status] != "Optimal":
        raise RuntimeError(f"LP solver failed with status: {LpStatus[status]}")

    values: List[float] = [value(var) for var in value_vars]  # type: ignore

    policy = []
    for state in range(mdp.num_states):
        if state in mdp.terminal_states:
            policy.append(0)
            continue

        action_values = [
            mdp.action_value(state, action, np.array(values))
            for action in range(mdp.num_actions)
        ]
        policy.append(int(np.argmax(action_values)))

    return values, policy


def evaluate_policy(
    mdp: MDP, policy: List[int]
) -> np.ndarray[Tuple[int], np.dtype[np.float64]]:
    A = np.zeros((mdp.num_states, mdp.num_states), dtype=np.float64)
    b = np.zeros(mdp.num_states, dtype=np.float64)
    for state in range(mdp.num_states):
        A[state, state] = 1.0

        if state in mdp.terminal_states:
            b[state] = 0.0
            continue

        for next_state in range(mdp.num_states):
            A[state, next_state] += (
                -mdp.gamma * mdp.transitions[state, policy[state], next_state]
            )
            b[state] += (
                mdp.transitions[state, policy[state], next_state]
                * mdp.rewards[state, policy[state], next_state]
            )

    return np.linalg.solve(A, b)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MDP planner")
    parser.add_argument("--mdp", required=True, help="Path to the input MDP file")
    parser.add_argument(
        "--algorithm",
        choices=["hpi", "lp"],
        default="hpi",
        help="Planning algorithm to use",
    )
    parser.add_argument(
        "--policy",
        help="Evaluate the supplied policy instead of computing an optimal one",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    mdp = MDP.from_file(args.mdp)

    if args.policy:
        policy = Policy.from_file(args.policy).state_to_action
        values = evaluate_policy(mdp, policy)
    else:
        if args.algorithm == "hpi":
            values, policy = solve_hpi(mdp)
        else:
            values, policy = solve_lp(mdp)

    for val, action in zip(values, policy):
        print(f"{val:.6f}\t{action}")
