# Retro Pixel Chess

A resizable Pygame chess app with local two-player and Stockfish modes, plus a DQN training experiment.

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

Press `M` to switch between local two-player and playing White against Stockfish. Press `R` to restart. Install Stockfish separately (for example, `brew install stockfish` on macOS) or set `STOCKFISH_PATH` to its executable.

For the playable mode, edit `STOCKFISH_SKILL_LEVEL` (0–20) and `STOCKFISH_MOVE_DELAY_MS` near the top of `game.py` to tune its strength and response pace.

## Train the DQN

```bash
python train.py --episodes 1000
tensorboard --logdir runs
```

Checkpoints are saved in `checkpoints/`. Resume from the most recent checkpoint with:

```bash
python train.py --resume checkpoints/latest.pt
```

TensorBoard reports episode reward, average loss, epsilon, moves survived, and win/draw/loss rates.
