"""Flatten Lichess's nested evaluation JSONL into ``fen,eval`` training CSV."""

import argparse
import csv
import json
import sys


def select_evaluation(position: dict, min_depth: int, mate_value: int):
    """Choose the highest-depth single-PV evaluation from one Lichess row."""
    candidates = [entry for entry in position.get("evals", [])
                  if entry.get("depth", 0) >= min_depth and entry.get("pvs")]
    if not candidates:
        return None
    pv = max(candidates, key=lambda entry: entry.get("depth", 0))["pvs"][0]
    if "cp" in pv:
        return pv["cp"]
    if "mate" in pv:
        return mate_value if pv["mate"] > 0 else -mate_value
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Lichess JSONL file, or '-' to read decompressed JSONL from stdin.")
    parser.add_argument("output", help="Output CSV path.")
    parser.add_argument("--limit", type=int, default=500_000)
    parser.add_argument("--min-depth", type=int, default=18)
    parser.add_argument("--mate-value", type=int, default=3000)
    args = parser.parse_args()
    source = sys.stdin if args.input == "-" else open(args.input, encoding="utf-8")
    written = 0
    try:
        with open(args.output, "w", newline="", encoding="utf-8") as target:
            writer = csv.writer(target)
            writer.writerow(("fen", "eval"))
            for line in source:
                try:
                    position = json.loads(line)
                    score = select_evaluation(position, args.min_depth, args.mate_value)
                    if score is not None:
                        writer.writerow((position["fen"], score))
                        written += 1
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue
                if args.limit and written >= args.limit:
                    break
    finally:
        if source is not sys.stdin:
            source.close()
    print(f"Wrote {written:,} positions to {args.output}")


if __name__ == "__main__":
    main()
