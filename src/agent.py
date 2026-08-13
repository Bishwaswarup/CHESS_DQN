"""Action-selection helpers for the Chess DQN."""

import random

import torch

from .chess_utils import action_index_to_move


def select_action(model, board, state, legal_mask, epsilon):
    """Return a legal move and its 0–4095 action index using epsilon-greedy play."""
    legal_moves = list(board.legal_moves)
    if random.random() < epsilon:
        move = random.choice(legal_moves)
        return move, move.from_square * 64 + move.to_square

    with torch.no_grad():
        action = model(state, mask=legal_mask).argmax().item()
    move = action_index_to_move(board, action)
    if move not in board.legal_moves:
        move = random.choice(legal_moves)
        action = move.from_square * 64 + move.to_square
    return move, action
