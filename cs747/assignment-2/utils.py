from collections import deque
from typing import List, Tuple


def hand_sum(state: Tuple[int, ...]) -> int:
    return sum((i + 1) * count for i, count in enumerate(state))


def generate_reachable_states(threshold: int) -> List[Tuple[int, ...]]:
    states = []
    empty_state = tuple([0] * 13)
    visited = {empty_state}
    queue = deque([empty_state])

    while queue:
        state = queue.popleft()

        # Skip if bust
        if hand_sum(state) >= threshold:
            continue

        states.append(state)

        for num in range(1, 14):
            available = 2 - state[num - 1]

            # Try adding
            if available > 0:
                new_state = list(state)
                new_state[num - 1] += 1
                new_state = tuple(new_state)

                if new_state not in visited:
                    visited.add(new_state)
                    queue.append(new_state)

            # Try swapping
            for swap_num in range(1, 14):
                if state[swap_num - 1] > 0 and available > 0 and swap_num != num:
                    new_state = list(state)
                    new_state[swap_num - 1] -= 1
                    new_state[num - 1] += 1
                    new_state = tuple(new_state)

                    if new_state not in visited:
                        visited.add(new_state)
                        queue.append(new_state)

    return states


def hand_to_state(hand: List[str]) -> Tuple[int, ...]:
    counts = [0] * 13
    for card in hand:
        num = int(card[:-1])
        counts[num - 1] += 1
    return tuple(counts)
