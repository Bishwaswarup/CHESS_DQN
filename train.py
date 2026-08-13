"""Train the DQN against Stockfish and record resumable, inspectable runs."""

import argparse
from pathlib import Path

import chess
import chess.engine
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from src.agent import select_action
from src.chess_utils import board_to_tensor, calculate_reward, device, get_legal_action_mask
from src.memory import ReplayBuffer
from src.model import ChessDQN
from src.utils import load_checkpoint, outcome_name, save_checkpoint


def train_step(model, optimizer, loss_fn, memory, batch_size=32, gamma=0.99):
    """Sample replay memory and perform one DQN update."""
    if len(memory) < batch_size:
        return 0.0

    states, actions, rewards, next_states, dones, masks = zip(*memory.sample(batch_size))
    states = torch.cat(states)
    next_states = torch.cat(next_states)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    actions = torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
    dones = torch.tensor(dones, dtype=torch.float32, device=device)
    next_masks = torch.cat(masks)

    current_q_values = model(states).gather(1, actions).squeeze(1)
    with torch.no_grad():
        next_q_values = model(next_states, mask=next_masks)
        max_next_q = next_q_values.max(1).values
        target_q_values = rewards + gamma * max_next_q * (1 - dones)

    loss = loss_fn(current_q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()


def run_training_loop(args):
    model = ChessDQN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.SmoothL1Loss()
    memory = ReplayBuffer(capacity=args.memory_size)
    start_episode, epsilon = 0, args.epsilon
    checkpoint_path = Path(args.checkpoint_dir) / "latest.pt"
    if args.resume:
        start_episode, epsilon = load_checkpoint(Path(args.resume), model, optimizer)
        print(f"Resumed checkpoint at episode {start_episode}.")

    writer = SummaryWriter(args.log_dir)
    outcomes = {"win": 0, "draw": 0, "loss": 0}
    stockfish_path = args.stockfish_path
    try:
        with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
            engine.configure({"Skill Level": args.stockfish_skill})
            for episode in range(start_episode + 1, args.episodes + 1):
                board = chess.Board()
                episode_reward, total_loss, updates, ai_moves = 0.0, 0.0, 0, 0

                while not board.is_game_over(claim_draw=True):
                    state = board_to_tensor(board)
                    mask = get_legal_action_mask(board)
                    move, action = select_action(model, board, state, mask, epsilon)

                    reward = calculate_reward(board, move)
                    board.push(move)
                    ai_moves += 1
                    if not board.is_game_over(claim_draw=True):
                        reply = engine.play(board, chess.engine.Limit(depth=args.stockfish_depth)).move
                        board.push(reply)

                    done = board.is_game_over(claim_draw=True)
                    if done:
                        result = outcome_name(board)
                        if result == "loss":
                            reward -= 100.0
                        elif result == "draw":
                            reward += 5.0
                    next_state = board_to_tensor(board)
                    next_mask = get_legal_action_mask(board)
                    memory.push(state, action, reward, next_state, done, next_mask)
                    episode_reward += reward
                    loss = train_step(model, optimizer, loss_fn, memory, args.batch_size)
                    total_loss += loss
                    updates += loss > 0

                result = outcome_name(board)
                outcomes[result] += 1
                epsilon = max(args.epsilon_min, epsilon * args.epsilon_decay)
                average_loss = total_loss / updates if updates else 0.0
                games = sum(outcomes.values())
                writer.add_scalar("training/episode_reward", episode_reward, episode)
                writer.add_scalar("training/average_loss", average_loss, episode)
                writer.add_scalar("training/epsilon", epsilon, episode)
                writer.add_scalar("game/ai_moves_survived", ai_moves, episode)
                writer.add_scalar("game/win_rate", outcomes["win"] / games, episode)
                writer.add_scalar("game/draw_rate", outcomes["draw"] / games, episode)
                writer.add_scalar("game/loss_rate", outcomes["loss"] / games, episode)
                print(f"Episode {episode}/{args.episodes} | reward {episode_reward:.1f} | "
                      f"moves {ai_moves} | {result} | loss {average_loss:.4f} | epsilon {epsilon:.3f}")

                if episode % args.save_every == 0 or episode == args.episodes:
                    save_checkpoint(checkpoint_path, model, optimizer, episode, epsilon)
                    save_checkpoint(Path(args.checkpoint_dir) / f"episode-{episode}.pt", model, optimizer, episode, epsilon)
    finally:
        writer.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--stockfish-path", default="stockfish")
    parser.add_argument("--stockfish-depth", type=int, default=5)
    parser.add_argument("--stockfish-skill", type=int, default=5, choices=range(21))
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--log-dir", default="runs/chess-dqn")
    parser.add_argument("--resume", help="Checkpoint path to resume from.")
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--memory-size", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--epsilon", type=float, default=0.5)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    return parser.parse_args()


if __name__ == "__main__":
    run_training_loop(parse_args())
