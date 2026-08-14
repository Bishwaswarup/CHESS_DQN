"""
Retro Pixel Chess
------------------
A minimalistic, Nintendo-inspired 2D chess game.

Controls:
    - Click a piece to select it (valid destinations light up)
    - Click a highlighted square to move there
    - Click the same piece again (or an empty/illegal square) to deselect
    - Press R at any time to restart the game

Rules implemented:
    - Full legal move generation for all pieces
    - Captures
    - Castling (kingside & queenside, with all legality checks)
    - En passant
    - Pawn promotion (auto-promotes to Queen)
    - Check, checkmate and stalemate detection
"""

import sys
import copy
import os
import shutil
from pathlib import Path
import pygame
import chess
import chess.engine

from src.engine import SearchEngine
from src.nnue import NNUEEvaluator

# ----------------------------------------------------------------------------
# Config / Palette (soft pastel retro look)
# ----------------------------------------------------------------------------

SQUARE = 64
BOARD_PX = SQUARE * 8
MARGIN_TOP = 64
MARGIN_BOTTOM = 40
WIDTH = BOARD_PX
HEIGHT = BOARD_PX + MARGIN_TOP + MARGIN_BOTTOM

# Pastel palette
COL_BG          = (238, 231, 221)
COL_LIGHT_SQ    = (240, 224, 214)   # pastel cream
COL_DARK_SQ     = (176, 196, 190)   # pastel sage
COL_SELECT      = (255, 214, 165)   # pastel orange highlight
COL_MOVE_DOT    = (150, 178, 170)   # muted teal dot
COL_CAPTURE_RING= (222, 150, 150)   # pastel red ring
COL_CHECK       = (235, 150, 150)   # pastel red king-in-check tint
COL_LAST_MOVE   = (210, 210, 170)   # soft highlight for last move
COL_TEXT        = (70, 65, 60)
COL_TEXT_SUB    = (120, 112, 104)
COL_WHITE_PIECE = (250, 248, 244)
COL_WHITE_EDGE  = (150, 140, 130)
COL_BLACK_PIECE = (80, 74, 70)
COL_BLACK_EDGE  = (40, 36, 34)
COL_PANEL       = (222, 214, 202)

FPS = 60
STOCKFISH_DEPTH = 8
STOCKFISH_SKILL_LEVEL = 10  # Range: 0 (weakest) to 20 (strongest).
STOCKFISH_MOVE_DELAY_MS = 750  # Pause after the engine has chosen a move.
SEARCH_DEPTH = 4
NNUE_CHECKPOINT = os.environ.get("NNUE_CHECKPOINT")  # Optional TinyNNUE checkpoint.

ASSET_DIR = Path(__file__).resolve().parent / "assets"
PIECE_NAMES = {"P": "Pawn", "R": "Rook", "N": "Knight", "B": "Bishop", "Q": "Queen", "K": "King"}

# ----------------------------------------------------------------------------
# Pixel-art sprite bitmaps (8x8 grid per piece). 'X' = filled pixel.
# Stylised NES-like silhouettes, reused for both colors (tinted differently).
# ----------------------------------------------------------------------------

SPRITES = {
    "P": [
        "...XX...",
        "..XXXX..",
        "...XX...",
        "..XXXX..",
        ".XXXXXX.",
        ".XXXXXX.",
        "XXXXXXXX",
        "XXXXXXXX",
    ],
    "R": [
        "X.X.X.XX",
        "XXXXXXXX",
        ".XXXXXX.",
        ".XXXXXX.",
        ".XXXXXX.",
        ".XXXXXX.",
        "XXXXXXXX",
        "XXXXXXXX",
    ],
    "N": [
        "...XXX..",
        "..XXXXX.",
        ".XXXXXXX",
        "XXXXXX..",
        "XXXXXXX.",
        "..XXXXX.",
        ".XXXXXXX",
        "XXXXXXXX",
    ],
    "B": [
        "...XX...",
        "..XXXX..",
        "...XX...",
        "..XXXX..",
        ".XXXXXX.",
        "..XXXX..",
        ".XXXXXX.",
        "XXXXXXXX",
    ],
    "Q": [
        "X.X.X.XX",
        ".XXXXXX.",
        "..XXXX..",
        ".XXXXXX.",
        ".XXXXXX.",
        ".XXXXXX.",
        "XXXXXXXX",
        "XXXXXXXX",
    ],
    "K": [
        "...XX...",
        "..XXXX..",
        "...XX...",
        ".XXXXXX.",
        "XXXXXXXX",
        ".XXXXXX.",
        ".XXXXXX.",
        "XXXXXXXX",
    ],
}

# ----------------------------------------------------------------------------
# Chess engine
# ----------------------------------------------------------------------------

FILES = "abcdefgh"

def in_bounds(r, c):
    return 0 <= r < 8 and 0 <= c < 8


class ChessGame:
    """Holds board state, turn, castling/en-passant rights, and rule logic."""

    def __init__(self):
        self.reset()

    def reset(self):
        # board[r][c] = None or (color, piece_letter); r=0 is rank8 (black back rank)
        self.board = [[None] * 8 for _ in range(8)]
        back = ["R", "N", "B", "Q", "K", "B", "N", "R"]
        for c in range(8):
            self.board[0][c] = ("b", back[c])
            self.board[1][c] = ("b", "P")
            self.board[6][c] = ("w", "P")
            self.board[7][c] = ("w", back[c])

        self.turn = "w"
        self.selected = None          # (r, c) of selected piece
        self.legal_targets = []       # list of (r, c, meta) legal destinations for selected
        self.last_move = None         # ((fr,fc),(tr,tc)) for highlight
        self.castling = {"w": {"K": True, "Q": True}, "b": {"K": True, "Q": True}}
        self.ep_target = None         # square eligible for en-passant capture this turn
        self.status = "White to move"
        self.game_over = False

    # ---------------- basic helpers ----------------

    def piece_at(self, r, c):
        return self.board[r][c]

    def find_king(self, color, board=None):
        board = board if board is not None else self.board
        for r in range(8):
            for c in range(8):
                p = board[r][c]
                if p and p[0] == color and p[1] == "K":
                    return (r, c)
        return None

    def opponent(self, color):
        return "b" if color == "w" else "w"

    # ---------------- attack detection ----------------

    def square_attacked(self, r, c, by_color, board=None):
        """Is square (r,c) attacked by any piece of by_color on given board?"""
        board = board if board is not None else self.board

        # Pawn attacks
        direction = 1 if by_color == "w" else -1  # pawn of by_color attacks "backwards" relative to its move
        for dc in (-1, 1):
            pr, pc = r + direction, c + dc
            if in_bounds(pr, pc):
                p = board[pr][pc]
                if p and p[0] == by_color and p[1] == "P":
                    return True

        # Knight attacks
        for dr, dc in [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]:
            pr, pc = r + dr, c + dc
            if in_bounds(pr, pc):
                p = board[pr][pc]
                if p and p[0] == by_color and p[1] == "N":
                    return True

        # King attacks (adjacency)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                pr, pc = r + dr, c + dc
                if in_bounds(pr, pc):
                    p = board[pr][pc]
                    if p and p[0] == by_color and p[1] == "K":
                        return True

        # Sliding: rook/queen (straight)
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            pr, pc = r + dr, c + dc
            while in_bounds(pr, pc):
                p = board[pr][pc]
                if p:
                    if p[0] == by_color and p[1] in ("R", "Q"):
                        return True
                    break
                pr += dr
                pc += dc

        # Sliding: bishop/queen (diagonal)
        for dr, dc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
            pr, pc = r + dr, c + dc
            while in_bounds(pr, pc):
                p = board[pr][pc]
                if p:
                    if p[0] == by_color and p[1] in ("B", "Q"):
                        return True
                    break
                pr += dr
                pc += dc

        return False

    def in_check(self, color, board=None):
        board = board if board is not None else self.board
        king_pos = self.find_king(color, board)
        if king_pos is None:
            return False
        return self.square_attacked(king_pos[0], king_pos[1], self.opponent(color), board)

    # ---------------- pseudo-legal move generation ----------------

    def pseudo_moves(self, r, c, board=None, castling=None, ep_target=None):
        board = board if board is not None else self.board
        castling = castling if castling is not None else self.castling
        ep_target = ep_target if ep_target is not None else self.ep_target

        p = board[r][c]
        if not p:
            return []
        color, kind = p
        moves = []  # (tr, tc, meta) meta describes special move

        if kind == "P":
            direction = -1 if color == "w" else 1
            start_row = 6 if color == "w" else 1
            promo_row = 0 if color == "w" else 7

            # forward one
            nr = r + direction
            if in_bounds(nr, c) and board[nr][c] is None:
                meta = "promo" if nr == promo_row else None
                moves.append((nr, c, meta))
                # forward two
                if r == start_row:
                    nr2 = r + 2 * direction
                    if board[nr2][c] is None:
                        moves.append((nr2, c, "double"))

            # captures
            for dc in (-1, 1):
                nr, nc = r + direction, c + dc
                if in_bounds(nr, nc):
                    target = board[nr][nc]
                    if target and target[0] != color:
                        meta = "promo" if nr == promo_row else None
                        moves.append((nr, nc, meta))
                    elif ep_target == (nr, nc) and target is None:
                        moves.append((nr, nc, "ep"))

        elif kind == "N":
            for dr, dc in [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]:
                nr, nc = r + dr, c + dc
                if in_bounds(nr, nc):
                    target = board[nr][nc]
                    if not target or target[0] != color:
                        moves.append((nr, nc, None))

        elif kind in ("R", "B", "Q"):
            dirs = []
            if kind in ("R", "Q"):
                dirs += [(-1,0),(1,0),(0,-1),(0,1)]
            if kind in ("B", "Q"):
                dirs += [(-1,-1),(-1,1),(1,-1),(1,1)]
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                while in_bounds(nr, nc):
                    target = board[nr][nc]
                    if target is None:
                        moves.append((nr, nc, None))
                    else:
                        if target[0] != color:
                            moves.append((nr, nc, None))
                        break
                    nr += dr
                    nc += dc

        elif kind == "K":
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if in_bounds(nr, nc):
                        target = board[nr][nc]
                        if not target or target[0] != color:
                            moves.append((nr, nc, None))

            # castling
            row = 7 if color == "w" else 0
            if r == row and c == 4 and not self.in_check(color, board):
                rights = castling[color]
                # kingside
                if rights["K"] and board[row][5] is None and board[row][6] is None:
                    rook = board[row][7]
                    if rook and rook[1] == "R" and rook[0] == color:
                        if not self.square_attacked(row, 5, self.opponent(color), board) and \
                           not self.square_attacked(row, 6, self.opponent(color), board):
                            moves.append((row, 6, "castleK"))
                # queenside
                if rights["Q"] and board[row][1] is None and board[row][2] is None and board[row][3] is None:
                    rook = board[row][0]
                    if rook and rook[1] == "R" and rook[0] == color:
                        if not self.square_attacked(row, 3, self.opponent(color), board) and \
                           not self.square_attacked(row, 2, self.opponent(color), board):
                            moves.append((row, 2, "castleQ"))

        return moves

    def simulate_move(self, r, c, tr, tc, meta, board, castling, ep_target):
        """Return a new (board, castling, ep_target) after applying the move."""
        board = copy.deepcopy(board)
        castling = copy.deepcopy(castling)
        p = board[r][c]
        color, kind = p
        captured_piece = board[tr][tc]
        new_ep = None

        board[r][c] = None
        board[tr][tc] = p

        if meta == "double":
            new_ep = ((r + tr) // 2, c)

        if meta == "ep":
            # captured pawn is on same row as origin, same col as destination
            board[r][tc] = None

        if meta == "promo":
            board[tr][tc] = (color, "Q")

        if meta == "castleK":
            row = tr
            board[row][5] = board[row][7]
            board[row][7] = None
        if meta == "castleQ":
            row = tr
            board[row][3] = board[row][0]
            board[row][0] = None

        if kind == "K":
            castling[color]["K"] = False
            castling[color]["Q"] = False
        if kind == "R":
            row = 7 if color == "w" else 0
            if r == row and c == 0:
                castling[color]["Q"] = False
            if r == row and c == 7:
                castling[color]["K"] = False
        # Capturing a rook on its original square also removes that castling right.
        if captured_piece and captured_piece[1] == "R":
            captured_color = captured_piece[0]
            captured_row = 7 if captured_color == "w" else 0
            if (tr, tc) == (captured_row, 0):
                castling[captured_color]["Q"] = False
            elif (tr, tc) == (captured_row, 7):
                castling[captured_color]["K"] = False
        return board, castling, new_ep

    def legal_moves_for(self, r, c):
        """Pseudo-legal moves filtered to those that don't leave own king in check."""
        p = self.board[r][c]
        if not p:
            return []
        color = p[0]
        result = []
        for (tr, tc, meta) in self.pseudo_moves(r, c):
            nb, ncastle, nep = self.simulate_move(r, c, tr, tc, meta, self.board, self.castling, self.ep_target)
            if not self.in_check(color, nb):
                result.append((tr, tc, meta))
        return result

    def all_legal_moves(self, color):
        moves = []
        for r in range(8):
            for c in range(8):
                p = self.board[r][c]
                if p and p[0] == color:
                    for m in self.legal_moves_for(r, c):
                        moves.append((r, c, m[0], m[1], m[2]))
        return moves

    # ---------------- move execution ----------------

    def make_move(self, r, c, tr, tc, meta):
        color = self.board[r][c][0]
        nb, ncastle, nep = self.simulate_move(r, c, tr, tc, meta, self.board, self.castling, self.ep_target)
        self.board = nb
        self.castling = ncastle
        self.ep_target = nep
        self.last_move = ((r, c), (tr, tc))
        self.turn = self.opponent(color)
        self.update_status()

    def update_status(self):
        color = self.turn
        moves = self.all_legal_moves(color)
        in_chk = self.in_check(color)
        name = "White" if color == "w" else "Black"
        if not moves:
            self.game_over = True
            if in_chk:
                winner = "Black" if color == "w" else "White"
                self.status = f"Checkmate — {winner} wins!"
            else:
                self.status = "Stalemate — Draw"
        else:
            self.game_over = False
            if in_chk:
                self.status = f"{name} to move — Check!"
            else:
                self.status = f"{name} to move"

    # ---------------- input handling ----------------

    def handle_click(self, r, c):
        if self.game_over:
            return False
        if not in_bounds(r, c):
            return False

        if self.selected is None:
            p = self.board[r][c]
            if p and p[0] == self.turn:
                self.selected = (r, c)
                self.legal_targets = self.legal_moves_for(r, c)
            return False

        sr, sc = self.selected
        if (r, c) == (sr, sc):
            self.selected = None
            self.legal_targets = []
            return False

        # clicked another own piece -> reselect
        p = self.board[r][c]
        if p and p[0] == self.turn:
            self.selected = (r, c)
            self.legal_targets = self.legal_moves_for(r, c)
            return False

        # attempt move
        for (tr, tc, meta) in self.legal_targets:
            if (tr, tc) == (r, c):
                self.make_move(sr, sc, tr, tc, meta)
                self.selected = None
                self.legal_targets = []
                return True

        # illegal target -> just deselect (clear feedback, no crash)
        self.selected = None
        self.legal_targets = []
        return False

    def to_fen(self):
        """Encode this game's state for Stockfish's standard UCI interface."""
        ranks = []
        for row in self.board:
            empty = 0
            rank = ""
            for piece in row:
                if piece is None:
                    empty += 1
                    continue
                if empty:
                    rank += str(empty)
                    empty = 0
                color, kind = piece
                rank += kind if color == "w" else kind.lower()
            ranks.append(rank + (str(empty) if empty else ""))

        rights = ""
        for color, label in (("w", "K"), ("w", "Q"), ("b", "K"), ("b", "Q")):
            if self.castling[color][label]:
                rights += label if color == "w" else label.lower()
        en_passant = "-" if self.ep_target is None else f"{FILES[self.ep_target[1]]}{8 - self.ep_target[0]}"
        return f"{'/'.join(ranks)} {'w' if self.turn == 'w' else 'b'} {rights or '-'} {en_passant} 0 1"

    def apply_uci_move(self, uci):
        """Apply a Stockfish UCI move through this game's normal move logic."""
        source, target = uci[:2], uci[2:4]
        r, c = 8 - int(source[1]), FILES.index(source[0])
        tr, tc = 8 - int(target[1]), FILES.index(target[0])
        for legal_tr, legal_tc, meta in self.legal_moves_for(r, c):
            if (legal_tr, legal_tc) == (tr, tc):
                self.make_move(r, c, tr, tc, meta)
                return True
        return False


# ----------------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------------

def load_piece_images():
    """Load the supplied piece PNGs once, after the display is initialised."""
    images = {}
    for color, side in (("w", "White"), ("b", "Black")):
        for kind, name in PIECE_NAMES.items():
            path = ASSET_DIR / f"Piece={name}, Side={side}.png"
            images[(color, kind)] = pygame.image.load(path).convert_alpha()
    return images


def board_layout(surface):
    """Return the board rectangle, keeping an 8x8 board centred when resized."""
    width, height = surface.get_size()
    board_size = min(width, max(0, height - MARGIN_TOP - MARGIN_BOTTOM))
    board_size -= board_size % 8  # integer-sized squares keep the board crisp
    board_x = (width - board_size) // 2
    return pygame.Rect(board_x, MARGIN_TOP, board_size, board_size)


def draw_board(surface, game, font, font_small, font_big, piece_images, mode, engine_available):
    surface.fill(COL_BG)
    width, height = surface.get_size()
    board_rect = board_layout(surface)
    square = board_rect.width // 8 if board_rect.width else 0

    # top status panel
    pygame.draw.rect(surface, COL_PANEL, (0, 0, width, MARGIN_TOP))
    status_surf = font_big.render(game.status, True, COL_TEXT)
    surface.blit(status_surf, (16, MARGIN_TOP // 2 - status_surf.get_height() // 2))

    engine_label = {"local": "Local two-player", "search": f"Built-in search (depth {SEARCH_DEPTH})",
                    "stockfish": "Stockfish"}[mode]
    availability = "" if engine_available or mode != "stockfish" else " (engine unavailable)"
    hint = f"{engine_label}{availability} | M: switch mode | R: restart"
    hint_surf = font_small.render(hint, True, COL_TEXT_SUB)
    surface.blit(hint_surf, (max(14, width - hint_surf.get_width() - 14), MARGIN_TOP - 18))

    if square == 0:
        return

    king_in_check_sq = None
    if game.in_check(game.turn) and not game.game_over:
        king_in_check_sq = game.find_king(game.turn)
    elif game.game_over and "Checkmate" in game.status:
        loser = game.turn
        king_in_check_sq = game.find_king(loser)

    for r in range(8):
        for c in range(8):
            x = board_rect.x + c * square
            y = board_rect.y + r * square
            base = COL_LIGHT_SQ if (r + c) % 2 == 0 else COL_DARK_SQ
            pygame.draw.rect(surface, base, (x, y, square, square))

            # last move highlight
            if game.last_move and (r, c) in game.last_move:
                s = pygame.Surface((square, square), pygame.SRCALPHA)
                s.fill((*COL_LAST_MOVE, 110))
                surface.blit(s, (x, y))

            # check tint
            if king_in_check_sq == (r, c):
                s = pygame.Surface((square, square), pygame.SRCALPHA)
                s.fill((*COL_CHECK, 130))
                surface.blit(s, (x, y))

            # selected highlight
            if game.selected == (r, c):
                s = pygame.Surface((square, square), pygame.SRCALPHA)
                s.fill((*COL_SELECT, 150))
                surface.blit(s, (x, y))

    # legal move markers
    for (tr, tc, meta) in game.legal_targets:
        x = board_rect.x + tc * square
        y = board_rect.y + tr * square
        occupied = game.board[tr][tc] is not None or meta == "ep"
        cx, cy = x + square // 2, y + square // 2
        if occupied:
            pygame.draw.circle(surface, COL_CAPTURE_RING, (cx, cy), max(2, square // 2 - 6), max(1, square // 16))
        else:
            pygame.draw.circle(surface, COL_MOVE_DOT, (cx, cy), max(3, square // 8))

    # pieces
    pad = max(2, square // 10)
    piece_size = square - pad * 2
    for r in range(8):
        for c in range(8):
            p = game.board[r][c]
            if p:
                x = board_rect.x + c * square + pad
                y = board_rect.y + r * square + pad
                image = pygame.transform.smoothscale(piece_images[(p[0], p[1])], (piece_size, piece_size))
                surface.blit(image, (x, y))

    # grid lines for crispness
    for i in range(9):
        pygame.draw.line(surface, (0, 0, 0, 20), (board_rect.x + i * square, board_rect.y),
                          (board_rect.x + i * square, board_rect.bottom), 1)
        pygame.draw.line(surface, (0, 0, 0, 20), (board_rect.x, board_rect.y + i * square),
                          (board_rect.right, board_rect.y + i * square), 1)

    # coordinate labels
    for c in range(8):
        lbl = font_small.render(FILES[c], True, COL_TEXT_SUB)
        surface.blit(lbl, (board_rect.x + c * square + 4, board_rect.y + board_rect.height - lbl.get_height() - 2))
    for r in range(8):
        lbl = font_small.render(str(8 - r), True, COL_TEXT_SUB)
        surface.blit(lbl, (board_rect.x + 4, board_rect.y + r * square + 3))

    # bottom panel
    pygame.draw.rect(surface, COL_PANEL, (0, board_rect.bottom, width, max(0, height - board_rect.bottom)))
    if game.game_over:
        restart_surf = font.render("Press R to play again", True, COL_TEXT)
        surface.blit(restart_surf, (width // 2 - restart_surf.get_width() // 2,
                                     board_rect.bottom + MARGIN_BOTTOM // 2 - restart_surf.get_height() // 2))


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------

def main():
    pygame.init()
    pygame.mixer.init()
    pygame.display.set_caption("Retro Pixel Chess")
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
    clock = pygame.time.Clock()
    piece_images = load_piece_images()
    move_sound = pygame.mixer.Sound(ASSET_DIR / "milanwulf-foot-switch-166326.mp3")

    try:
        font = pygame.font.SysFont("Courier New", 18, bold=True)
        font_small = pygame.font.SysFont("Courier New", 13)
        font_big = pygame.font.SysFont("Courier New", 22, bold=True)
    except Exception:
        font = pygame.font.Font(None, 22)
        font_small = pygame.font.Font(None, 16)
        font_big = pygame.font.Font(None, 26)

    game = ChessGame()
    mode = "local"
    try:
        local_engine = SearchEngine(NNUEEvaluator(NNUE_CHECKPOINT) if NNUE_CHECKPOINT else None)
    except (OSError, RuntimeError, KeyError) as error:
        local_engine = SearchEngine()
        print(f"Could not load NNUE checkpoint; using hand evaluation: {error}")
    stockfish = None
    engine_error = None
    stockfish_path = os.environ.get("STOCKFISH_PATH") or shutil.which("stockfish")

    def start_stockfish():
        nonlocal stockfish, engine_error
        if stockfish is not None:
            return True
        if not stockfish_path:
            engine_error = "Stockfish not found. Set STOCKFISH_PATH or install Stockfish."
            return False
        try:
            stockfish = chess.engine.SimpleEngine.popen_uci(stockfish_path)
            stockfish.configure({"Skill Level": STOCKFISH_SKILL_LEVEL})
            engine_error = None
            return True
        except (OSError, chess.engine.EngineError) as error:
            engine_error = f"Could not start Stockfish: {error}"
            return False

    def reset_game():
        game.reset()
        if mode == "stockfish" and not start_stockfish():
            game.status = engine_error

    def play_stockfish_turn():
        if mode != "stockfish" or game.turn != "b" or game.game_over or not start_stockfish():
            return False
        game.status = "Stockfish is thinking..."
        draw_board(screen, game, font, font_small, font_big, piece_images, mode, stockfish is not None)
        pygame.display.flip()
        try:
            result = stockfish.play(chess.Board(game.to_fen()), chess.engine.Limit(depth=STOCKFISH_DEPTH))
            if result.move and game.apply_uci_move(result.move.uci()):
                # Give the player a short, intentional pause before the reply appears.
                pygame.time.delay(STOCKFISH_MOVE_DELAY_MS)
                move_sound.play()
                return True
        except chess.engine.EngineError as error:
            game.status = f"Stockfish error: {error}"
        return False

    def play_search_turn():
        if mode != "search" or game.turn != "b" or game.game_over:
            return False
        game.status = "Built-in engine is thinking..."
        draw_board(screen, game, font, font_small, font_big, piece_images, mode, True)
        pygame.display.flip()
        result = local_engine.search(chess.Board(game.to_fen()), max_depth=SEARCH_DEPTH)
        if result.move and game.apply_uci_move(result.move.uci()):
            pygame.time.delay(STOCKFISH_MOVE_DELAY_MS)
            move_sound.play()
            return True
        return False

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    reset_game()
                elif event.key == pygame.K_m:
                    mode = {"local": "search", "search": "stockfish", "stockfish": "local"}[mode]
                    reset_game()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                layout = board_layout(screen)
                if layout.collidepoint(mx, my) and layout.width:
                    square = layout.width // 8
                    c = (mx - layout.x) // square
                    r = (my - layout.y) // square
                    if game.handle_click(r, c):
                        move_sound.play()
                        play_search_turn() or play_stockfish_turn()

        draw_board(screen, game, font, font_small, font_big, piece_images, mode, stockfish is not None)
        pygame.display.flip()
        clock.tick(FPS)

    if stockfish is not None:
        stockfish.quit()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
