import chess
import torch

def select_device():
    """Use the fastest available PyTorch backend: CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


device = select_device()

# Convert the chess into a 3d shape of
# 12 channels (each piece has a channel that is pawn,knight,bishop,rook,queen,king so 6 pieces and 2 colors)
# 8 rows and 8 columns

def board_to_tensor(board: chess.Board) -> torch.Tensor:
    tensor = torch.zeros((12, 8, 8),  dtype=torch.float32, device = device)

    piece_idx = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }

    for square, piece in board.piece_map().items():
        row = 7 - (square//8)
        column = square%8
        channel = piece_idx[piece.piece_type] + (0 if piece.color == chess.WHITE else 6)
        tensor[channel, row, column] = 1.0

    return tensor.unsqueeze(0)

def get_legal_action_mask(board: chess.Board) -> torch.Tensor:
    # Create a 4096 element mask
    mask = torch.zeros(4096, dtype=torch.float32, device=device)
    for move in board.legal_moves:
        action_idx = move.from_square * 64 + move.to_square
        mask[action_idx] = 1.0
    return mask.unsqueeze(0)


def action_index_to_move(board: chess.Board, action_idx: int) -> chess.Move:
    """Converts a chosen action integer index back to a chess.Move."""
    from_sq = action_idx // 64
    to_sq = action_idx % 64
    move = chess.Move(from_sq, to_sq)

    # Handle pawn promotion default to Queen [cite: 8]
    if board.piece_at(from_sq) and board.piece_at(from_sq).piece_type == chess.PAWN:
        if (to_sq >= 56 or to_sq <= 7):
            move.promotion = chess.QUEEN

    return move


PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
}


def material_score(board: chess.Board) -> float:
    """Material balance from White's perspective (positive means White is ahead)."""
    score = 0.0
    for piece in board.piece_map().values():
        value = PIECE_VALUES.get(piece.piece_type, 0.0)
        score += value if piece.color == chess.WHITE else -value
    return score


def calculate_transition_reward(before: chess.Board, after: chess.Board) -> float:
    """Reward net material after both sides have moved, then score the final result.

    Comparing full turns teaches the agent that a tempting capture is bad when
    Stockfish can immediately recapture a more valuable piece.
    """
    reward = (material_score(after) - material_score(before)) * 0.2
    outcome = after.outcome(claim_draw=True)
    if outcome is not None:
        if outcome.winner == chess.WHITE:
            reward += 1.0
        elif outcome.winner == chess.BLACK:
            reward -= 1.0
        else:
            reward += 0.5
    return max(-1.0, min(1.0, reward))
