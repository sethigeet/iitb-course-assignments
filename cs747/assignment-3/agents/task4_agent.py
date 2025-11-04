import numpy as np

from minichess.chess.fastchess import Chess
from minichess.chess.fastchess_utils import bit_count, piece_matrix_to_legal_moves

from .base_agent import BaseAgent

# fmt: off

# Game ending scenarios
CHECKMATE_SCORE = 1000000
DRAW_SCORE = 0

# Bonuses
CHECK_BONUS = 40
PIN_BONUS = 30
MOBILITY_BONUS = 2
ROOK_OPEN_FILE_BONUS = 25

PIECE_VALUES = {
    0: 100,  # Pawn
    1: 300,  # Knight
    2: 300,  # Bishop
    3: 500,  # Rook
    4: 900,  # Queen
    5: 0,    # King
}
PIECE_VALUES_ARRAY = np.array([100, 300, 300, 500, 900, 0])

# Incentivized to advance
PAWN_VALUE_BY_POSITION = np.array([
    [0,   0,   0,   0],   # Row 0 (Start)
    [10,  10,  10,  10],  # Row 1
    [25,  25,  25,  25],  # Row 2
    [60,  60,  60,  60],  # Row 3
    [100, 100, 100, 100]  # Row 4 (Promotion)
])

# Incentivized to control the center
KNIGHT_VALUE_BY_POSITION = np.array([
    [0, 10, 10, 0],    # Row 0
    [10, 25, 25, 10],  # Row 1
    [10, 25, 25, 10],  # Row 2
    [10, 25, 25, 10],  # Row 3
    [0, 10, 10, 0]     # Row 4
])

# Incentivized to control the center
BISHOP_VALUE_BY_POSITION = np.array([
    [5, 10, 10, 5],    # Row 0
    [10, 20, 20, 10],  # Row 1
    [10, 20, 20, 10],  # Row 2
    [10, 20, 20, 10],  # Row 3
    [5, 10, 10, 5]     # Row 4
])

# Incentivized to control the center
ROOK_VALUE_BY_POSITION = np.array([
    [5, 10, 10, 5],    # Row 0
    [5, 10, 10, 5],    # Row 1
    [5, 10, 10, 5],    # Row 2
    [5, 10, 10, 5],    # Row 3
    [5, 10, 10, 5]     # Row 4
])

# Incentivized to control the center
QUEEN_VALUE_BY_POSITION = np.array([
    [0, 5,  5,  0],    # Row 0
    [5, 10, 10, 5],    # Row 1
    [5, 10, 10, 5],    # Row 2
    [5, 10, 10, 5],    # Row 3
    [0, 5,  5,  0]     # Row 4
])

# Incentivized to stay on the back rank for safety
KING_VALUE_BY_POSITION = np.array([
    [10, 20, 20, 10],  # Row 0
    [0,  5,  5,  0],   # Row 1
    [0,  0,  0,  0],   # Row 2
    [0,  0,  0,  0],   # Row 3
    [0,  0,  0,  0]    # Row 4
])

PIECE_VALUE_BY_POSITION = {
    0: PAWN_VALUE_BY_POSITION,
    1: KNIGHT_VALUE_BY_POSITION,
    2: BISHOP_VALUE_BY_POSITION,
    3: ROOK_VALUE_BY_POSITION,
    4: QUEEN_VALUE_BY_POSITION,
    5: KING_VALUE_BY_POSITION,
}

# fmt: on


def evaluate_board(board: Chess, original_turn: int):
    # NOTE: We always evaluate the board from the perspective of the original turn
    # player since this is handled appropriately in the minimax function

    white_multiplier = 1 if original_turn == 1 else -1
    black_multiplier = 1 if original_turn == 0 else -1

    # Check if the game is over
    result = board.game_result()
    if result is not None:
        if result == 1:
            return white_multiplier * CHECKMATE_SCORE
        elif result == -1:
            return black_multiplier * CHECKMATE_SCORE
        else:
            return DRAW_SCORE

    score = 0

    # Add scores for all pieces on the board
    score += (
        white_multiplier
        * PIECE_VALUES_ARRAY[
            board.piece_lookup[1, :, :][board.piece_lookup[1, :, :] != -1]
        ].sum()
    )
    score += (
        black_multiplier
        * PIECE_VALUES_ARRAY[
            board.piece_lookup[0, :, :][board.piece_lookup[0, :, :] != -1]
        ].sum()
    )

    # Add bonus for piece location
    for piece_type in range(6):
        white_piece_positions = np.where(board.piece_lookup[1, :, :] == piece_type)
        score += (
            white_multiplier
            * (
                PIECE_VALUE_BY_POSITION[piece_type][
                    board.dims[0] - 1 - white_piece_positions[0],
                    white_piece_positions[1],
                ]
            ).sum()
        )

        black_piece_positions = np.where(board.piece_lookup[0, :, :] == piece_type)
        score += (
            black_multiplier
            * (
                PIECE_VALUE_BY_POSITION[piece_type][
                    black_piece_positions[0], black_piece_positions[1]
                ]
            ).sum()
        )

    all_pieces = board.get_all_pieces(False)
    white_pieces = board.get_all_pieces(False, [1])
    white_king_pos = board.find_king(1)
    black_pieces = board.get_all_pieces(False, [0])
    black_king_pos = board.find_king(0)

    # Add bonus for check
    score += (
        white_multiplier
        * CHECK_BONUS
        * bit_count(board.find_checkers(all_pieces, 0, white_king_pos))
    )
    score += (
        black_multiplier
        * CHECK_BONUS
        * bit_count(board.find_checkers(all_pieces, 1, black_king_pos))
    )

    # Add bonus for pinned pieces
    white_pinned_pieces = board.find_pinned_pieces(
        all_pieces, black_pieces, 0, white_king_pos
    )
    black_pinned_pieces = board.find_pinned_pieces(
        all_pieces, white_pieces, 1, black_king_pos
    )
    score -= (
        white_multiplier
        * PIN_BONUS
        * len(white_pinned_pieces[white_pinned_pieces != np.iinfo(np.uint64).max])
    )
    score -= (
        black_multiplier
        * PIN_BONUS
        * len(black_pinned_pieces[black_pinned_pieces != np.iinfo(np.uint64).max])
    )

    # Add bonus for mobility
    num_legal_moves = len(piece_matrix_to_legal_moves(*board.legal_moves()))
    if board.turn == 1:
        score += white_multiplier * MOBILITY_BONUS * num_legal_moves
    else:
        score += black_multiplier * MOBILITY_BONUS * num_legal_moves

    # Bust the cache to get the correct mobility bonus for the other turn
    board.legal_move_cache, board.promotion_move_cache = None, None
    board.turn = 1 - board.turn
    num_legal_moves = len(piece_matrix_to_legal_moves(*board.legal_moves()))
    if board.turn == 1:
        score += white_multiplier * MOBILITY_BONUS * num_legal_moves
    else:
        score += black_multiplier * MOBILITY_BONUS * num_legal_moves

    # Reset the turn to the previous turn
    board.turn = 1 - board.turn
    # Re-calculate the legal moves to get back the correct cache
    board.legal_moves()

    return score


class Task4Agent(BaseAgent):
    rng = np.random.default_rng(8228)
    max_depth = 4
    cache = {}

    def __init__(self, name="Task4Agent"):
        super().__init__(name)

    def move(self, chess_obj: Chess):
        best_move, _ = self.minimax(
            board=chess_obj,
            depth=self.max_depth,
            alpha=float("-inf"),
            beta=float("inf"),
            maximizing=True,
            original_turn=chess_obj.turn,
        )
        if best_move is None:
            legal_moves = piece_matrix_to_legal_moves(*chess_obj.legal_moves())
            best_move = legal_moves[self.rng.choice(range(len(legal_moves)))]

        return best_move

    def minimax(
        self,
        board: Chess,
        depth: int,
        alpha: float,
        beta: float,
        maximizing: bool,
        original_turn: int,
    ):
        board_hash_key = (
            board.agent_board_state().tobytes(),
            depth,
            maximizing,
            original_turn,
        )
        if board_hash_key in self.cache:
            return self.cache[board_hash_key]

        legal_moves = piece_matrix_to_legal_moves(*board.legal_moves())

        if depth == 0 or not legal_moves or board.game_result() is not None:
            return None, evaluate_board(
                board, board.turn if maximizing else 1 - board.turn
            )

        # Shuffle moves for slightly different games and potentially
        # better pruning if move ordering is poor.
        np.random.shuffle(legal_moves)

        best_move = None
        if maximizing:
            max_score = float("-inf")
            for move_tuple in legal_moves:
                (i, j), (dx, dy), promotion = move_tuple
                temp_board = board.copy()
                temp_board.make_move(i, j, dx, dy, promotion)
                _, new_score = self.minimax(
                    temp_board, depth - 1, alpha, beta, False, original_turn
                )
                if new_score > max_score:
                    max_score = new_score
                    best_move = move_tuple

                alpha = max(alpha, max_score)
                if beta <= alpha:
                    break

            self.cache[board_hash_key] = (best_move, max_score)
            return best_move, max_score
        else:
            min_score = float("inf")
            for move_tuple in legal_moves:
                (i, j), (dx, dy), promotion = move_tuple
                temp_board = board.copy()
                temp_board.make_move(i, j, dx, dy, promotion)
                _, new_score = self.minimax(
                    temp_board, depth - 1, alpha, beta, True, original_turn
                )
                if new_score < min_score:
                    min_score = new_score
                    best_move = move_tuple

                beta = min(beta, min_score)
                if beta <= alpha:
                    break

            self.cache[board_hash_key] = (best_move, min_score)
            return best_move, min_score
