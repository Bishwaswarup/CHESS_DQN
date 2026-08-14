"""Small sparse piece-square evaluator and checkpoint adapter.

This is NNUE-style (piece-square features + accumulator + MLP), not a claim
of Stockfish NNUE compatibility.  It is intentionally small enough to be used
inside a Python searcher and can replace ``hand_evaluate`` after training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import chess
import torch
from torch import nn

FEATURES = 12 * 64


def feature_indices(board: chess.Board) -> list[int]:
    return [((piece.piece_type - 1) + (0 if piece.color else 6)) * 64 + square
            for square, piece in board.piece_map().items()]


class TinyNNUE(nn.Module):
    """Sparse first layer; 256 hidden nodes is a practical Python baseline."""
    def __init__(self, hidden_size: int = 256):
        super().__init__()
        self.hidden_size = hidden_size
        self.feature_weights = nn.Embedding(FEATURES, hidden_size)
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.output = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward_indices(self, batch: Iterable[list[int]]) -> torch.Tensor:
        accumulators = []
        for indices in batch:
            if indices:
                index = torch.tensor(indices, dtype=torch.long, device=self.bias.device)
                accumulators.append(self.feature_weights(index).sum(dim=0) + self.bias)
            else:
                accumulators.append(self.bias)
        return self.output(torch.relu(torch.stack(accumulators))).squeeze(-1)


class NNUEEvaluator:
    """Callable adapter returning a White-perspective centipawn score."""
    def __init__(self, checkpoint: str | Path, device: str = "cpu"):
        data = torch.load(checkpoint, map_location=device, weights_only=True)
        self.model = TinyNNUE(data.get("hidden_size", 256)).to(device)
        self.model.load_state_dict(data["state_dict"])
        self.model.eval()

    @torch.no_grad()
    def __call__(self, board: chess.Board) -> int:
        return int(self.model.forward_indices([feature_indices(board)]).item())
