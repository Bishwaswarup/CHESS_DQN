"""Checkpoint and result helpers shared by training scripts."""

import chess
import torch

from .chess_utils import device


def save_checkpoint(path, model, optimizer, episode, epsilon):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "episode": episode,
        "epsilon": epsilon,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)


def load_checkpoint(path, model, optimizer):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint.get("episode", 0), checkpoint.get("epsilon", 0.5)


def outcome_name(board):
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return "draw"
    return "win" if outcome.winner == chess.WHITE else "loss"
