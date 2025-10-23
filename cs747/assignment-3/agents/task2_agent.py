import random

from minichess.chess.fastchess import Chess
from minichess.chess.fastchess_utils import piece_matrix_to_legal_moves

from .base_agent import BaseAgent


class Task2Agent(BaseAgent):
    def __init__(self, name="Task2Agent"):
        super().__init__(name)

    def move(self, chess_obj: Chess):
        ### Your code goes here ###
        """right now the behaves similar to a random agent, picking moves uniformly randomly, change it to
        design your agent
        """
        moves, proms = chess_obj.legal_moves()
        legal_moves = piece_matrix_to_legal_moves(moves, proms)
        move = random.choice(legal_moves)
        return move

    ### Any other utility functions you want to define for your agent.
    def reset(
        self,
    ): ...
