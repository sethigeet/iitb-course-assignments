import argparse
from typing import Tuple

from utils import generate_reachable_states, hand_sum


class GameConfig:
    threshold: int
    bonus: int
    sequence: Tuple[int, int, int]

    def __init__(self, threshold: int, bonus: int, sequence: Tuple[int, int, int]):
        self.threshold = threshold
        self.bonus = bonus
        self.sequence = sequence

    @staticmethod
    def from_file(path: str) -> "GameConfig":
        with open(path, "r") as f:
            lines = [line.strip() for line in f.readlines()]

        threshold = int(lines[1])
        bonus = int(lines[2])
        sequence = tuple(map(int, lines[3].split()))
        assert len(sequence) == 3

        return GameConfig(threshold, bonus, sequence)

    def get_bonus(self, state: Tuple[int, ...]) -> int:
        if all(state[num - 1] > 0 for num in self.sequence):
            return self.bonus
        return 0


def encode_mdp(game_config: GameConfig):
    states = generate_reachable_states(game_config.threshold)

    # Create state index mapping
    state_to_idx = {state: idx for idx, state in enumerate(states)}
    num_states = len(states)
    num_actions = 15  # 0-14

    terminal_state_idx = num_states
    num_states += 1

    # Print MDP header
    print(f"numStates {num_states}")
    print(f"numActions {num_actions}")
    print(f"end {terminal_state_idx}")

    transitions = []
    for state_idx, state in enumerate(states):
        num_remaining = 26 - sum(state)

        # No cards left => can only stop
        if num_remaining == 0:
            action = 14
            reward = hand_sum(state) + game_config.get_bonus(state)
            transitions.append((state_idx, action, terminal_state_idx, reward, 1.0))
            continue

        # Generate transitions for all actions
        for action in range(num_actions):
            # Add
            if action == 0:
                # Try adding each possible card number
                prob_of_exceeding_threshold = 0.0
                for num in range(1, 14):
                    available = 2 - state[num - 1]
                    if available == 0:
                        continue

                    prob = available / num_remaining

                    # New state after adding
                    new_state = list(state)
                    new_state[num - 1] += 1
                    new_state = tuple(new_state)

                    if hand_sum(new_state) >= game_config.threshold:
                        prob_of_exceeding_threshold += prob
                        continue

                    next_state_idx = state_to_idx[new_state]
                    transitions.append((state_idx, action, next_state_idx, 0.0, prob))

                if prob_of_exceeding_threshold > 0.0:
                    transitions.append(
                        (
                            state_idx,
                            action,
                            terminal_state_idx,
                            0.0,
                            prob_of_exceeding_threshold,
                        )
                    )

            # Stop
            elif action == 14:
                reward = hand_sum(state) + game_config.get_bonus(state)
                transitions.append((state_idx, action, terminal_state_idx, reward, 1.0))

            # Swap (actions 1-13)
            else:
                swap_num = action

                # Check if we have this number in hand
                if state[swap_num - 1] == 0:  # Invalid action
                    transitions.append(
                        (state_idx, action, terminal_state_idx, 0.0, 1.0)
                    )
                    continue

                # Try drawing each possible card number
                prob_of_exceeding_threshold = 0.0
                for draw_num in range(1, 14):
                    available = 2 - state[draw_num - 1]
                    if available == 0:
                        continue

                    prob = available / num_remaining

                    # New state after swapping
                    new_state = list(state)
                    new_state[swap_num - 1] -= 1
                    new_state[draw_num - 1] += 1
                    new_state = tuple(new_state)

                    if hand_sum(new_state) >= game_config.threshold:
                        prob_of_exceeding_threshold += prob
                        continue

                    next_state_idx = state_to_idx[new_state]
                    transitions.append((state_idx, action, next_state_idx, 0.0, prob))

                if prob_of_exceeding_threshold > 0.0:
                    transitions.append(
                        (
                            state_idx,
                            action,
                            terminal_state_idx,
                            0.0,
                            prob_of_exceeding_threshold,
                        )
                    )

    # Print all transitions
    for s, a, s_next, r, p in transitions:
        print(f"transition {s} {a} {s_next} {r} {p}")

    # Print MDP type and gamma
    print("mdptype episodic")
    print("gamma 1.0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode card game as MDP")
    parser.add_argument(
        "--game_config", required=True, help="Path to game configuration file"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    game_config = GameConfig.from_file(args.game_config)
    encode_mdp(game_config)
