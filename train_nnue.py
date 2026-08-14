"""Supervised training for TinyNNUE from CSV or JSONL FEN/evaluation data."""

import argparse
import csv
import json
import random
from pathlib import Path

import chess
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from src.nnue import TinyNNUE, feature_indices


class EvaluationDataset(Dataset):
    def __init__(self, path: str, fen_column: str, eval_column: str, limit: int | None = None):
        rows = []
        with open(path, encoding="utf-8") as handle:
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

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def collate(batch):
    features, targets = zip(*batch)
    return list(features), torch.tensor(targets, dtype=torch.float32)


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    """Shared loop for both training and validation passes.

    Passing an optimizer runs a training epoch (with backprop); omitting it
    runs a no-grad validation epoch. Keeping both in one function guarantees
    the loss is computed identically in both modes, so the numbers are
    actually comparable.
    """
    training = optimizer is not None
    model.train(mode=training)
    total, count = 0.0, 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for features, targets in loader:
            targets = targets.to(device).clamp(-3_000, 3_000)
            predictions = model.forward_indices(features)
            loss = loss_fn(predictions, targets)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * len(targets)
            count += len(targets)
    return total / count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", help="CSV or JSONL with a FEN and centipawn evaluation column.")
    parser.add_argument("--fen-column", default="fen")
    parser.add_argument("--eval-column", default="eval")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--val-fraction", type=float, default=0.1,
                        help="Fraction of positions held out for validation (not trained on).")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the train/val split.")
    parser.add_argument("--output", default="checkpoints/tiny_nnue.pt",
                        help="Where to save the checkpoint with the best validation loss.")
    parser.add_argument("--last-output",
                        help="Optional path to also save the final-epoch checkpoint, "
                             "even if it wasn't the best on validation.")
    parser.add_argument("--resume",
                         help="Path to a checkpoint to continue training from. "
                              "Hidden size must match --hidden-size.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    dataset = EvaluationDataset(args.data, args.fen_column, args.eval_column, args.limit)

    indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(indices)
    val_size = max(1, int(len(indices) * args.val_fraction))
    val_indices, train_indices = indices[:val_size], indices[val_size:]
    train_set, val_set = Subset(dataset, train_indices), Subset(dataset, val_indices)
    print(f"{len(train_set)} training positions, {len(val_set)} validation positions "
          f"({args.val_fraction:.0%} held out, seed {args.seed}).")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = TinyNNUE(args.hidden_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = torch.nn.SmoothL1Loss()

    best_val_loss = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        if checkpoint["hidden_size"] != args.hidden_size:
            raise ValueError(f"Checkpoint hidden_size={checkpoint['hidden_size']} does not match "
                              f"--hidden-size={args.hidden_size}.")
        model.load_state_dict(checkpoint["state_dict"])
        # Only trust the checkpoint's val_loss as a starting "best" if it was
        # evaluated on the same held-out split we're about to use; otherwise
        # treat this as a fresh comparison so a lucky old split can't block
        # saving a genuinely better model on this run's split.
        if checkpoint.get("val_loss") is not None and checkpoint.get("seed") == args.seed \
                and checkpoint.get("val_fraction") == args.val_fraction:
            best_val_loss = checkpoint["val_loss"]
        print(f"Resumed weights from {args.resume} "
              f"(epoch {checkpoint.get('epoch', '?')}, val Huber {checkpoint.get('val_loss', float('nan')):.2f} cp).")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_loss = run_epoch(model, val_loader, loss_fn, device, optimizer=None)
        flag = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"hidden_size": args.hidden_size, "state_dict": model.state_dict(),
                        "epoch": epoch, "val_loss": val_loss,
                        "seed": args.seed, "val_fraction": args.val_fraction}, output)
            flag = "  <- saved (best val so far)"
        print(f"epoch {epoch}/{args.epochs}: train Huber {train_loss:.2f} cp | "
              f"val Huber {val_loss:.2f} cp{flag}")

    if args.last_output:
        last_output = Path(args.last_output)
        last_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"hidden_size": args.hidden_size, "state_dict": model.state_dict(),
                    "epoch": args.epochs, "val_loss": val_loss,
                    "seed": args.seed, "val_fraction": args.val_fraction}, last_output)
        print(f"Saved final-epoch checkpoint to {last_output}")

    print(f"Best checkpoint: {output} (val Huber {best_val_loss:.2f} cp)")


if __name__ == "__main__":
    main()