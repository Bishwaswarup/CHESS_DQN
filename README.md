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
python train.py --episodes 1000 --stockfish=1
tensorboard --logdir runs
```

Checkpoints are saved in `checkpoints/`. Resume from the most recent checkpoint with:

```bash
python train.py --resume checkpoints/latest.pt
```

TensorBoard reports episode reward, average loss, epsilon, moves survived, and win/draw/loss rates.

`--stockfish` sets Stockfish's skill from `0` (weakest) to `20` (strongest). Start a new DQN at `--stockfish=1 --stockfish-depth=1`; raise these gradually after its greedy evaluation results improve. The trainer uses reward clipping and a target network to keep Q-values stable.

## Google Colab

In Colab, select **Runtime → Change runtime type → T4 GPU**, then run:

```bash
!git clone https://github.com/Bishwaswarup/CHESS_DQN.git
%cd CHESS_DQN
!pip install -r requirements.txt
!sudo apt-get update -qq && sudo apt-get install -y stockfish
!python train.py --episodes 1000 --stockfish=1 --stockfish-depth=1 --stockfish-path /usr/games/stockfish
```

The code automatically selects CUDA in Colab, MPS on supported Macs, and CPU elsewhere. To continue a prior Colab session, store `checkpoints/` in Google Drive, then pass its `latest.pt` file to `--resume`.
