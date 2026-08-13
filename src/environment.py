"""Chess environment wrapper used by training code and future experiments."""

import chess

from .chess_utils import board_to_tensor, get_legal_action_mask


class ChessEnvironment:
    """A White-playing environment with a python-chess board as its state."""

    def __init__(self):
        self.board = chess.Board()

    def reset(self):
        self.board.reset()
        return self.observation()

    def observation(self):
        return board_to_tensor(self.board), get_legal_action_mask(self.board)

    def is_done(self):
        return self.board.is_game_over(claim_draw=True)
