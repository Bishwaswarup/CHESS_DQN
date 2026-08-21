# ♟️ CHESS_training

A fully playable Python chess application with a built-in alpha-beta search engine and an NNUE (Efficiently Updatable Neural Network Evaluation) trainer pipeline — trained on real Lichess evaluation data.

---

## Features

### Playable Game (`game.py`)
- Resizable Pygame GUI with piece graphics and sound effects
- **Three modes** (toggle with `M`):
  - Local two-player
  - Built-in alpha-beta engine (configurable depth)
  - Stockfish engine (requires Stockfish binary)
- Full chess rules: castling, en passant, promotion, draw detection

### Search Engine (`src/engine.py`)
- Iterative deepening alpha-beta pruning
- Move ordering heuristics (MVV-LVA, killer moves)
- Transposition table with Zobrist hashing
- Evaluation: material count + piece-square tables

### NNUE Trainer (`train_nnue.py`)
- Sparse neural network evaluator trained via supervised learning
- Distills Stockfish evaluations into a compact model
- Position features encoded as piece-square occupancy vectors

### Data Pipeline
| Script | Purpose |
|---|---|
| `prepare_lichess_evals.py` | Extract positions from Lichess database dumps |
| `label_with_stockfish.py` | Annotate positions with Stockfish centipawn scores |
| `benchmark.py` | Compare engine strength against Stockfish |

---

## Installation

```bash
git clone https://github.com/Bishwaswarup/CHESS_training.git
cd CHESS_training
pip install -r requirements.txt
```

> **Optional:** Download a [Stockfish binary](https://stockfishchess.org/download/) and place it in the project root for Stockfish mode.

---

## Usage

```bash
# Play the game
python game.py

# Train the neural evaluator (requires pre-labelled data)
python train_nnue.py

# Prepare training data from a Lichess PGN/eval dump
python prepare_lichess_evals.py

# Benchmark the engine
python benchmark.py
```

---

## Tech Stack

- **Python 3.10+**
- [Pygame](https://www.pygame.org/) — GUI and game loop
- [NumPy](https://numpy.org/) — Position encoding and training
- [python-chess](https://python-chess.readthedocs.io/) — Move generation and board representation
- Stockfish (optional) — Data labelling and engine comparison

---

## Project Structure

```
CHESS_training/
├── assets/             # Piece sprites and sound effects
├── src/
│   ├── engine.py       # Alpha-beta search + evaluation
│   ├── board.py        # Board representation
│   └── ...
├── game.py             # Main Pygame application
├── train_nnue.py       # Neural evaluator training
├── prepare_lichess_evals.py
├── label_with_stockfish.py
├── benchmark.py
└── requirements.txt
```

---

## Background

NNUE (Efficiently Updatable Neural Network Evaluation) was pioneered in the Stockfish project. This implementation reproduces a simplified version — a sparse, incrementally-updated neural network that maps board features to centipawn evaluations — combined with a classical alpha-beta tree search for move selection.

---

## License

MIT
