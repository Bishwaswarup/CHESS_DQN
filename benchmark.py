"""Benchmark the local engine against a transparent Stockfish strength ladder."""

import argparse
import csv
import shutil
from pathlib import Path

import chess
import chess.engine

from src.engine import SearchEngine

LADDER = ((0, 1), (3, 3), (8, 6))


def play_game(ours: SearchEngine, sf: chess.engine.SimpleEngine, our_depth: int, sf_depth: int, our_white: bool,
              max_plies: int) -> str:
    board = chess.Board()
    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        if board.turn == our_white:
            result = ours.search(board, max_depth=our_depth)
            if result.move is None:
                break
            board.push(result.move)
        else:
            board.push(sf.play(board, chess.engine.Limit(depth=sf_depth)).move)
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return "draw"
    return "win" if outcome.winner == our_white else "loss"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stockfish-path", default=shutil.which("stockfish") or "stockfish")
    parser.add_argument("--games", type=int, default=4, help="Games per ladder rung; colours alternate.")
    parser.add_argument("--our-depth", type=int, default=4)
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument("--nnue-checkpoint")
    parser.add_argument("--output", default="benchmarks/ladder.csv")
    args = parser.parse_args()
    # Keep the hand-evaluation benchmark runnable with only python-chess.
    # Torch is needed only when a learned checkpoint is explicitly requested.
    evaluator = None
    if args.nnue_checkpoint:
        from src.nnue import NNUEEvaluator
        evaluator = NNUEEvaluator(args.nnue_checkpoint)
    ours = SearchEngine(evaluator=evaluator)
    rows = []
    with chess.engine.SimpleEngine.popen_uci(args.stockfish_path) as sf:
        for skill, sf_depth in LADDER:
            sf.configure({"Skill Level": skill})
            scores = {"win": 0, "draw": 0, "loss": 0}
            for game in range(args.games):
                scores[play_game(ours, sf, args.our_depth, sf_depth, game % 2 == 0, args.max_plies)] += 1
            total = sum(scores.values())
            row = {"skill": skill, "stockfish_depth": sf_depth, "our_depth": args.our_depth,
                   **scores, "score_rate": (scores["win"] + 0.5 * scores["draw"]) / total}
            rows.append(row)
            print(f"skill {skill:2d}, SF depth {sf_depth}: {scores['win']}W/{scores['draw']}D/{scores['loss']}L "
                  f"({row['score_rate']:.1%})")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
