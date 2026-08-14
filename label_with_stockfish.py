"""Create TinyNNUE supervision data by evaluating positions from a PGN with Stockfish."""

import argparse
import csv
from pathlib import Path

import chess
import chess.engine
import chess.pgn


def score_to_cp(score: chess.engine.Score, mate_value: int) -> int:
    """Normalise Stockfish's White-relative score, including forced mates."""
    return score.white().score(mate_score=mate_value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pgn", help="A standard-chess PGN file to sample positions from.")
    parser.add_argument("--stockfish-path", default="stockfish")
    parser.add_argument("--output", default="evaluations.csv")
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--every", type=int, default=4, help="Evaluate every N plies in each game.")
    parser.add_argument("--limit", type=int, default=100_000)
    parser.add_argument("--mate-value", type=int, default=3000)
    args = parser.parse_args()
    if args.every < 1:
        parser.error("--every must be at least 1")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(args.pgn, encoding="utf-8", errors="replace") as games, \
            output.open("w", newline="", encoding="utf-8") as target, \
            chess.engine.SimpleEngine.popen_uci(args.stockfish_path) as engine:
        writer = csv.writer(target)
        writer.writerow(("fen", "eval"))
        while written < args.limit:
            game = chess.pgn.read_game(games)
            if game is None:
                break
            if game.headers.get("Variant", "Standard") not in ("Standard", "From Position"):
                continue
            board = game.board()
            for ply, move in enumerate(game.mainline_moves(), start=1):
                board.push(move)
                if ply % args.every:
                    continue
                info = engine.analyse(board, chess.engine.Limit(depth=args.depth))
                writer.writerow((board.fen(), score_to_cp(info["score"], args.mate_value)))
                written += 1
                if written >= args.limit:
                    break
    print(f"Stockfish labelled {written:,} positions in {output}")


if __name__ == "__main__":
    main()
