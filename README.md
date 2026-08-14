# Retro Pixel Chess

A resizable Pygame chess app with local two-player, a built-in alpha-beta engine,
and Stockfish modes, plus supervised TinyNNUE training.

## Project layout

```text
chess/
├── assets/                 # Piece images and move sound used by game.py
├── src/
│   ├── __init__.py
│   ├── engine.py           # Alpha-beta searcher and hand evaluation
│   └── nnue.py             # Sparse learned evaluator
├── game.py
├── train_nnue.py
├── benchmark.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Run the game

```bash
python -m pip install -r requirements.txt
python game.py
```

Press `M` to cycle local two-player → playing White against the built-in searcher →
Stockfish. Press `R` to restart. Install Stockfish separately (for example,
`brew install stockfish` on macOS) or set `STOCKFISH_PATH` to its executable.

For the playable mode, edit `STOCKFISH_SKILL_LEVEL` (0–20) and `STOCKFISH_MOVE_DELAY_MS` near the top of `game.py` to tune its strength and response pace.

## Built-in search engine

`src/engine.py` is the baseline to improve: iterative deepening alpha-beta,
Zobrist-keyed transposition table, MVV-LVA / killer / history move ordering,
and capture quiescence search. Its default evaluation is deliberately cheap
(material, piece-square terms and bishop pair) so search depth stays the main
source of strength.

Run a transparent progress benchmark, alternating colours at every rung:

```bash
python3 benchmark.py --stockfish-path /path/to/stockfish --games 12 --our-depth 4
```

This writes `benchmarks/ladder.csv` with W/D/L and score rate against Stockfish
skill/depth `(0, 1)`, `(3, 3)`, and `(8, 6)`. Increase game count before making
strength claims; 12 games is only a smoke-test-sized sample.

## Train and use TinyNNUE

`train_nnue.py` trains a small sparse piece-square network by distilling a
table of FEN positions and centipawn evaluations. Provide CSV columns `fen` and
`eval` (or JSONL fields with those names); evaluation should be from White's
perspective. This keeps ingestion explicit, so different Lichess/Stockfish dump
formats can be converted without silently training on the wrong perspective.

```bash
python3 train_nnue.py evaluations.csv --epochs 8 --output checkpoints/tiny_nnue.pt
NNUE_CHECKPOINT=checkpoints/tiny_nnue.pt python3 game.py
python3 benchmark.py --nnue-checkpoint checkpoints/tiny_nnue.pt --games 24
```

The checkpoint only replaces the static evaluator; alpha-beta, ordering and
quiescence are unchanged. Benchmark the hand-eval checkpoint and NNUE checkpoint
at the same search depth before increasing network size. This makes it clear
whether a model actually improves playing strength rather than merely slowing
the search.

## Google Colab

In Colab, select **Runtime → Change runtime type → T4 GPU**, then run:

```bash
!git clone https://github.com/Bishwaswarup/CHESS_DQN.git
%cd CHESS_DQN
!pip install -r requirements.txt
!sudo apt-get update -qq && sudo apt-get install -y stockfish
!python3 train_nnue.py evaluations.csv --epochs 12 --batch-size 1024 --output checkpoints/tiny_nnue.pt
!python3 benchmark.py --stockfish-path /usr/games/stockfish --nnue-checkpoint checkpoints/tiny_nnue.pt --games 24
```

The NNUE trainer automatically selects CUDA in Colab, MPS on supported Macs, and CPU elsewhere. To continue a prior Colab session, store `evaluations.csv` and `checkpoints/` in Google Drive.
