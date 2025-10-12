import argparse
from typing import Dict, List, Tuple

from utils import generate_reachable_states, hand_to_state


class TestCase:
    threshold: int
    bonus: int
    sequence: Tuple[int, int, int]
    hands: List[List[str]]

    def __init__(
        self,
        threshold: int,
        bonus: int,
        sequence: Tuple[int, int, int],
        hands: List[List[str]],
    ):
        self.threshold = threshold
        self.bonus = bonus
        self.sequence = sequence
        self.hands = hands

    @staticmethod
    def from_file(path: str) -> "TestCase":
        with open(path, "r") as f:
            lines = [line.strip() for line in f.read().strip().split("\n")]

        threshold = int(lines[1])
        bonus = int(lines[2])
        sequence = tuple(map(int, lines[3].split()))
        assert len(sequence) == 3

        hands = []
        for i in range(5, len(lines)):
            if not lines[i]:
                hands.append([])
                continue

            cards_str = lines[i].split()
            hands.append(cards_str)
        return TestCase(threshold, bonus, sequence, hands)


class Policy:
    policy: List[int]

    def __init__(self, policy: List[int]):
        self.policy = policy

    @staticmethod
    def from_file(path: str) -> "Policy":
        policy = []

        with open(path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                parts = line.strip().split()
                policy.append(int(parts[1]))

        return Policy(policy)


def get_optimal_action(
    hand: List[str], state_to_idx: Dict[Tuple[int, ...], int], policy: List[int]
) -> int:
    state = hand_to_state(hand)

    if state not in state_to_idx:
        raise ValueError(f"Hand {hand} (state {state}) not found in state space")

    state_idx = state_to_idx[state]
    action = policy[state_idx]

    if action == 0:
        return 0
    elif action == 14:
        return 27

    # Check if we have this specific card
    heart_card, diamond_card = f"{action}H", f"{action}D"
    if heart_card in hand:
        return action
    elif diamond_card in hand:
        return action + 13
    else:
        return -1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode policy for card game test cases"
    )
    parser.add_argument(
        "--value_policy", required=True, help="Path to value-policy file from planner"
    )
    parser.add_argument("--testcase", required=True, help="Path to test case file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Parse inputs
    test_case = TestCase.from_file(args.testcase)
    policy = Policy.from_file(args.value_policy).policy

    # Generate state space (same as encoder)
    states = generate_reachable_states(test_case.threshold)
    state_to_idx = {state: idx for idx, state in enumerate(states)}

    # For each test hand, output the optimal action
    for test_hand in test_case.hands:
        action = get_optimal_action(test_hand, state_to_idx, policy)
        print(action)
