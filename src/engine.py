"""A compact, deterministic chess searcher built on :mod:`python-chess`.

The engine deliberately prioritises nodes per second: a cheap evaluation is
searched deeply, while a learned evaluator can be plugged in once trained.
Scores are centipawns from the side-to-move perspective inside negamax.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import random
import time
from typing import Callable, Optional

import chess


INF = 100_000
MATE = 90_000
MATE_BOUND = MATE - 1_000
MAX_PLY = 128
PIECE_VALUE = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
               chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 20_000}

# Values are intentionally modest: tactical decisions should come from search,
# not from an over-fitted static evaluator.
PST = {
    chess.PAWN: (0, 5, 5, 0, 5, 10, 50, 0),
    chess.KNIGHT: (-50, -40, -30, -30, -30, -30, -40, -50),
    chess.BISHOP: (-20, -10, -10, -10, -10, -10, -10, -20),
    chess.ROOK: (0, 0, 0, 5, 5, 0, 0, 0),
    chess.QUEEN: (-20, -10, -10, -5, -5, -10, -10, -20),
    chess.KING: (20, 30, 10, 0, 0, 10, 30, 20),
}


class Bound(IntEnum):
    EXACT = 0
    LOWER = 1
    UPPER = 2


@dataclass(slots=True)
class TTEntry:
    depth: int
    score: int
    bound: Bound
    move: Optional[chess.Move]


@dataclass(slots=True)
class SearchResult:
    move: Optional[chess.Move]
    score: int
    depth: int
    nodes: int
    elapsed: float
    pv: list[chess.Move]


class ZobristHasher:
    """Stable position hashes independent of python-chess private internals."""

    def __init__(self, seed: int = 0xC0DEC0DE):
        rng = random.Random(seed)
        self.pieces = [[[rng.getrandbits(64) for _ in chess.SQUARES]
                        for _ in range(6)] for _ in range(2)]
        self.turn = rng.getrandbits(64)
        self.castling = [rng.getrandbits(64) for _ in range(16)]
        self.ep_file = [rng.getrandbits(64) for _ in range(8)]

    def hash(self, board: chess.Board) -> int:
        key = 0
        for square, piece in board.piece_map().items():
            key ^= self.pieces[int(piece.color)][piece.piece_type - 1][square]
        if board.turn == chess.BLACK:
            key ^= self.turn
        rights = 0
        rights |= int(board.has_kingside_castling_rights(chess.WHITE))
        rights |= int(board.has_queenside_castling_rights(chess.WHITE)) << 1
        rights |= int(board.has_kingside_castling_rights(chess.BLACK)) << 2
        rights |= int(board.has_queenside_castling_rights(chess.BLACK)) << 3
        key ^= self.castling[rights]
        # An EP square that cannot be captured has no effect on legal moves.
        if board.has_legal_en_passant():
            key ^= self.ep_file[chess.square_file(board.ep_square)]
        return key


def hand_evaluate(board: chess.Board) -> int:
    """Return a small material + piece-square score from White's viewpoint."""
    score = 0
    for square, piece in board.piece_map().items():
        row = chess.square_rank(square)
        # PST uses rank only to remain deliberately cheap.  Mirror it for Black.
        pst = PST[piece.piece_type][row if piece.color else 7 - row]
        value = PIECE_VALUE[piece.piece_type] + pst
        score += value if piece.color == chess.WHITE else -value
    # A bishop pair is a useful low-cost positional signal.
    if len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2:
        score += 30
    if len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2:
        score -= 30
    return score


class SearchEngine:
    """Iterative-deepening negamax with TT, killers, history and quiescence."""

    def __init__(self, evaluator: Optional[Callable[[chess.Board], int]] = None,
                 tt_size: int = 500_000):
        self.evaluator = evaluator or hand_evaluate
        self.tt_size = tt_size
        self.hasher = ZobristHasher()
        self.tt: dict[int, TTEntry] = {}
        self.killers: list[list[Optional[chess.Move]]] = [[None, None] for _ in range(MAX_PLY)]
        self.history = [[0 for _ in chess.SQUARES] for _ in chess.SQUARES]
        self.nodes = 0
        self.deadline: Optional[float] = None
        self.stop = False

    def clear(self) -> None:
        self.tt.clear()
        self.history = [[0 for _ in chess.SQUARES] for _ in chess.SQUARES]
        self.killers = [[None, None] for _ in range(MAX_PLY)]

    def search(self, board: chess.Board, max_depth: int = 5,
               time_limit: Optional[float] = None) -> SearchResult:
        """Find a move without changing ``board``. ``time_limit`` is seconds."""
        if board.is_game_over(claim_draw=True):
            return SearchResult(None, self._terminal_score(board, 0), 0, 0, 0.0, [])
        started = time.perf_counter()
        self.deadline = started + time_limit if time_limit else None
        self.nodes, self.stop = 0, False
        best_move, best_score, completed = None, -INF, 0
        for depth in range(1, max_depth + 1):
            score, move = self._root(board, depth)
            if self.stop:
                break
            best_move, best_score, completed = move, score, depth
            if abs(score) >= MATE_BOUND:
                break
        elapsed = time.perf_counter() - started
        return SearchResult(best_move, best_score, completed, self.nodes, elapsed,
                            self.principal_variation(board))

    def _root(self, board: chess.Board, depth: int) -> tuple[int, Optional[chess.Move]]:
        alpha, beta = -INF, INF
        best_score, best_move = -INF, None
        key = self.hasher.hash(board)
        tt_move = self.tt.get(key).move if key in self.tt else None
        for move in self._ordered_moves(board, tt_move, 0):
            board.push(move)
            score = -self._negamax(board, depth - 1, -beta, -alpha, 1)
            board.pop()
            if self.stop:
                return 0, None
            if score > best_score:
                best_score, best_move = score, move
            alpha = max(alpha, score)
        if best_move is not None:
            self._store(key, TTEntry(depth, best_score, Bound.EXACT, best_move))
        return best_score, best_move

    def _negamax(self, board: chess.Board, depth: int, alpha: int, beta: int, ply: int) -> int:
        self.nodes += 1
        if self.nodes & 2047 == 0 and self.deadline and time.perf_counter() >= self.deadline:
            self.stop = True
        if self.stop:
            return 0
        if board.is_game_over(claim_draw=True):
            return self._terminal_score(board, ply)
        if depth <= 0:
            return self._quiescence(board, alpha, beta, ply)

        key, alpha_start, beta_start = self.hasher.hash(board), alpha, beta
        entry = self.tt.get(key)
        tt_move = entry.move if entry else None
        if entry and entry.depth >= depth:
            if entry.bound == Bound.EXACT:
                return entry.score
            if entry.bound == Bound.LOWER:
                alpha = max(alpha, entry.score)
            else:
                beta = min(beta, entry.score)
            if alpha >= beta:
                return entry.score

        best_move, best_score = None, -INF
        for move in self._ordered_moves(board, tt_move, ply):
            board.push(move)
            score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1)
            board.pop()
            if self.stop:
                return 0
            if score > best_score:
                best_score, best_move = score, move
            if score > alpha:
                alpha = score
            if alpha >= beta:
                if not board.is_capture(move):
                    self._record_quiet_cutoff(move, depth, ply)
                break
        bound = Bound.UPPER if best_score <= alpha_start else Bound.LOWER if best_score >= beta_start else Bound.EXACT
        self._store(key, TTEntry(depth, best_score, bound, best_move))
        return best_score

    def _quiescence(self, board: chess.Board, alpha: int, beta: int, ply: int) -> int:
        self.nodes += 1
        if board.is_game_over(claim_draw=True):
            return self._terminal_score(board, ply)
        stand_pat = self._relative_eval(board)
        if stand_pat >= beta:
            return beta
        alpha = max(alpha, stand_pat)
        # A checked side must consider all evasions; otherwise capture-only is enough.
        moves = list(board.legal_moves) if board.is_check() else [m for m in board.legal_moves if board.is_capture(m)]
        for move in self._ordered_moves(board, None, ply, moves):
            board.push(move)
            score = -self._quiescence(board, -beta, -alpha, ply + 1)
            board.pop()
            if score >= beta:
                return beta
            alpha = max(alpha, score)
        return alpha

    def _ordered_moves(self, board: chess.Board, tt_move: Optional[chess.Move], ply: int,
                       moves=None) -> list[chess.Move]:
        moves = list(board.legal_moves) if moves is None else moves
        def score(move: chess.Move) -> int:
            if move == tt_move:
                return 10_000_000
            if board.is_capture(move):
                victim = board.piece_at(move.to_square)
                if victim is None:  # en passant
                    victim_value = PIECE_VALUE[chess.PAWN]
                else:
                    victim_value = PIECE_VALUE[victim.piece_type]
                attacker = board.piece_at(move.from_square)
                return 1_000_000 + 16 * victim_value - PIECE_VALUE[attacker.piece_type]
            if move == self.killers[ply][0]:
                return 900_000
            if move == self.killers[ply][1]:
                return 800_000
            return self.history[move.from_square][move.to_square]
        return sorted(moves, key=score, reverse=True)

    def _record_quiet_cutoff(self, move: chess.Move, depth: int, ply: int) -> None:
        if move != self.killers[ply][0]:
            self.killers[ply][1] = self.killers[ply][0]
            self.killers[ply][0] = move
        self.history[move.from_square][move.to_square] += depth * depth

    def _relative_eval(self, board: chess.Board) -> int:
        score = int(self.evaluator(board))
        return score if board.turn == chess.WHITE else -score

    def _terminal_score(self, board: chess.Board, ply: int) -> int:
        if board.is_checkmate():
            return -MATE + ply
        return 0

    def _store(self, key: int, entry: TTEntry) -> None:
        if len(self.tt) >= self.tt_size and key not in self.tt:
            # Python dicts do not expose a cheap LRU; discarding a small batch
            # keeps lookup fast without allocating a replacement structure.
            for old_key in list(self.tt)[: max(1, self.tt_size // 20)]:
                del self.tt[old_key]
        current = self.tt.get(key)
        if current is None or entry.depth >= current.depth:
            self.tt[key] = entry

    def principal_variation(self, board: chess.Board, max_length: int = 16) -> list[chess.Move]:
        probe = board.copy(stack=False)
        pv = []
        for _ in range(max_length):
            entry = self.tt.get(self.hasher.hash(probe))
            if not entry or not entry.move or entry.move not in probe.legal_moves:
                break
            pv.append(entry.move)
            probe.push(entry.move)
        return pv
