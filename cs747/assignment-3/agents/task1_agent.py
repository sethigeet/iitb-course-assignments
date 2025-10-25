import numpy as np

from minichess.chess.fastchess import Chess
from minichess.chess.fastchess_utils import piece_matrix_to_legal_moves

from .base_agent import BaseAgent

PIECE_VALUES = {
    0: 100,  # Pawn
    1: 300,  # Knight
    2: 300,  # Bishop
    3: 500,  # Rook
    4: 900,  # Queen
    5: 0,  # King
}
CHECKMATE_SCORE = 1000000
PAWN_PROMOTION_BONUS = 10
CHECK_BONUS = 40
PIN_BONUS = 25
DRAW_SCORE = 0


def evaluate_board(board: Chess):
    # Check if the game is over
    result = board.game_result()
    if result is not None:
        if result == 1:
            # NOTE: This has the be the other way around since this function is
            # called after the game is over so the last move was the player who
            # led to the game ending
            return -CHECKMATE_SCORE if board.turn == 1 else CHECKMATE_SCORE
        elif result == -1:
            return -CHECKMATE_SCORE if board.turn == 0 else CHECKMATE_SCORE
        else:
            return DRAW_SCORE

    # NOTE: We always evaluate the board from the perspective of the white
    # player since this is handled appropriately in the minimax function

    score = 0

    # Add scores for all pieces on the board
    for i in range(5):
        for j in range(4):
            piece_type, piece_color = board.any_piece_at(i, j)  # type: ignore
            if piece_type != -1:
                if piece_color == 1:
                    score += PIECE_VALUES[piece_type]
                    if piece_type == 0:
                        score += PAWN_PROMOTION_BONUS * (board.dims[0] - 1 - i)
                else:
                    score -= PIECE_VALUES[piece_type]
                    if piece_type == 0:
                        score -= PAWN_PROMOTION_BONUS * (board.dims[0] - 1 - i)

    # all_pieces = board.get_all_pieces(False)
    # white_pieces = board.get_all_pieces(False, [1])
    # white_king_pos = board.find_king(1)
    # black_pieces = board.get_all_pieces(False, [0])
    # black_king_pos = board.find_king(0)

    # Add bonus for check
    # score -= CHECK_BONUS * bit_count(board.find_checkers(all_pieces, 0, white_king_pos))
    # score += CHECK_BONUS * bit_count(board.find_checkers(all_pieces, 1, black_king_pos))

    # Add bonus for pinned pieces
    # white_pinned_pieces = board.find_pinned_pieces(
    #     all_pieces, black_pieces, 0, white_king_pos
    # )
    # black_pinned_pieces = board.find_pinned_pieces(
    #     all_pieces, white_pieces, 1, black_king_pos
    # )
    # for i in range(5):
    #     for j in range(4):
    #         if white_pinned_pieces[i, j] != np.iinfo(np.uint64).max:
    #             score -= PIN_BONUS
    #         if black_pinned_pieces[i, j] != np.iinfo(np.uint64).max:
    #             score += PIN_BONUS

    return score


class Task1Agent(BaseAgent):
    rng = np.random.default_rng(8228)
    max_depth = 2

    def __init__(self, name="Task1Agent"):
        super().__init__(name)

    def move(self, chess_obj: Chess):
        moves, proms = chess_obj.legal_moves()
        legal_moves = piece_matrix_to_legal_moves(moves, proms)

        # If there are no legal moves, choose a random move
        if not legal_moves:
            return legal_moves[self.rng.choice(range(len(legal_moves)))]

        best_move = None
        best_score = float("-inf")
        for move_tuple in legal_moves:
            (i, j), (dx, dy), promotion = move_tuple

            new_board = chess_obj.copy()
            new_board.make_move(i, j, dx, dy, promotion)
            new_score = self.minimax(
                new_board, depth=self.max_depth - 1, maximizing=False
            )
            if new_score > best_score:
                best_score = new_score
                best_move = move_tuple

        if best_move is None:
            best_move = legal_moves[self.rng.choice(range(len(legal_moves)))]

        return best_move

    def minimax(self, board: Chess, depth: int, maximizing: bool):
        if depth == 0 or not board.has_legal_moves or board.game_result() is not None:
            return evaluate_board(board)

        piece_matrix, promo_matrix = board.legal_moves()
        legal_moves = piece_matrix_to_legal_moves(piece_matrix, promo_matrix)

        if maximizing:
            max_score = float("-inf")
            for move_tuple in legal_moves:
                (i, j), (dx, dy), promotion = move_tuple
                temp_board = board.copy()
                temp_board.make_move(i, j, dx, dy, promotion)
                new_score = self.minimax(temp_board, depth - 1, False)
                max_score = max(max_score, new_score)
            return max_score
        else:
            min_score = float("inf")
            for move_tuple in legal_moves:
                (i, j), (dx, dy), promotion = move_tuple
                temp_board = board.copy()
                temp_board.make_move(i, j, dx, dy, promotion)
                new_score = self.minimax(temp_board, depth - 1, True)
                min_score = min(min_score, new_score)
            return min_score
