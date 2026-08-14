# Retro Pixel Chess

A resizable Pygame chess app with local two-player, a built-in alpha-beta engine,
and Stockfish modes, plus DQN and supervised NNUE experiments.

## Project layout

```text
chess-dqn/
├── assets/                 # Piece images and move sound used by game.py
├── src/
│   ├── __init__.py
│   ├── model.py            # Existing DQN model
│   ├── chess_utils.py      # Existing board/tensor helpers
│   ├── memory.py           # Existing replay buffer
│   ├── environment.py
│   ├── agent.py
│   └── utils.py
├── notebooks/
│   └── colab_training.ipynb
├── game.py
├── train.py
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

## Train the DQN

```bash
python train.py --episodes 1000 --stockfish=1
tensorboard --logdir runs
```

Checkpoints are saved in `checkpoints/`. Resume from the most recent checkpoint with:

```bash
python train.py --resume checkpoints/latest.pt
```

TensorBoard reports episode reward, average loss, epsilon, moves survived, and win/draw/loss rates.

`--stockfish` sets Stockfish's skill from `0` (weakest) to `20` (strongest). Start a new DQN at `--stockfish=1 --stockfish-depth=1`; raise these gradually after its greedy evaluation results improve. The trainer uses material-balance rewards after both sides move, Double DQN, dueling value/advantage heads, and prioritized replay to focus on costly mistakes.

The current board encoder uses 18 planes: the 12 piece planes, side to move, four castling-rights planes, and an en-passant plane. Because this changes the model input shape, start a **fresh training run** after this update; old checkpoints cannot be loaded.

For a learning curriculum, first train basic tactical play against random legal moves, then continue that checkpoint against Stockfish:

```bash
python train.py --episodes 3000 --opponent random --checkpoint-dir checkpoints-random
python train.py --episodes 5000 --stockfish=1 --stockfish-depth=1 \
  --resume checkpoints-random/latest.pt --checkpoint-dir checkpoints-stockfish
```

The default training settings batch `256` experiences every `4` game moves for efficient T4 GPU use. The replay buffer deliberately lives in system RAM; its job is to store many past games, while GPU memory is reserved for model batches. Priorities and Q-targets are capped to prevent a few outlier games from destabilising training.

## Google Colab

In Colab, select **Runtime → Change runtime type → T4 GPU**, then run:

```bash
!git clone https://github.com/Bishwaswarup/CHESS_DQN.git
%cd CHESS_DQN
!pip install -r requirements.txt
!sudo apt-get update -qq && sudo apt-get install -y stockfish
!python train.py --episodes 3000 --opponent random --checkpoint-dir checkpoints-random
# Then fine-tune the saved model against weak Stockfish:
!python train.py --episodes 5000 --stockfish=1 --stockfish-depth=1 --stockfish-path /usr/games/stockfish --resume checkpoints-random/latest.pt --checkpoint-dir checkpoints-stockfish
```

The code automatically selects CUDA in Colab, MPS on supported Macs, and CPU elsewhere. To continue a prior Colab session, store `checkpoints/` in Google Drive, then pass its `latest.pt` file to `--resume`.
