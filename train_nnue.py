"""Supervised training for TinyNNUE from CSV or JSONL FEN/evaluation data."""

import argparse
import csv
import json
from pathlib import Path

import chess
import torch
from torch.utils.data import DataLoader, Dataset

from src.nnue import TinyNNUE, feature_indices


class EvaluationDataset(Dataset):
    def __init__(self, path: str, fen_column: str, eval_column: str, limit: int | None = None):
        rows = []
        opener = open
        with opener(path, encoding="utf-8") as handle:
            reader = (json.loads(line) for line in handle) if path.endswith(".jsonl") else csv.DictReader(handle)
            for row in reader:
                try:
                    rows.append((feature_indices(chess.Board(row[fen_column])), float(row[eval_column])))
                except (KeyError, ValueError):
                    continue
                if limit and len(rows) >= limit:
                    break
        if not rows:
            raise ValueError("No valid positions found; check --fen-column and --eval-column.")
        self.rows = rows

    def __len__(self): return len(self.rows)
    def __getitem__(self, index): return self.rows[index]


def collate(batch):
    features, targets = zip(*batch)
    return list(features), torch.tensor(targets, dtype=torch.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", help="CSV or JSONL with a FEN and centipawn evaluation column.")
    parser.add_argument("--fen-column", default="fen")
    parser.add_argument("--eval-column", default="eval")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", default="checkpoints/tiny_nnue.pt")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    dataset = EvaluationDataset(args.data, args.fen_column, args.eval_column, args.limit)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    model = TinyNNUE(args.hidden_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = torch.nn.SmoothL1Loss()
    for epoch in range(1, args.epochs + 1):
        total = 0.0
        for features, targets in loader:
            targets = targets.to(device).clamp(-3_000, 3_000)
            loss = loss_fn(model.forward_indices(features), targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(targets)
        print(f"epoch {epoch}/{args.epochs}: Huber {total / len(dataset):.2f} cp")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"hidden_size": args.hidden_size, "state_dict": model.cpu().state_dict()}, output)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
