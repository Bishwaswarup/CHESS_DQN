"""Train the DQN against Stockfish and record resumable, inspectable runs."""

import argparse
import random
from contextlib import nullcontext
from pathlib import Path

import chess
import chess.engine
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from src.agent import select_action
from src.chess_utils import board_to_tensor, calculate_transition_reward, device, get_legal_action_mask
from src.memory import ReplayBuffer
from src.model import ChessDQN
from src.utils import load_checkpoint, outcome_name, save_checkpoint


def train_step(model, target_model, optimizer, loss_fn, memory, batch_size=32, gamma=0.99,
               beta=0.4, q_value_clip=10.0):
    """Sample replay memory and perform one DQN update."""
    if len(memory) < batch_size:
        return 0.0

    batch, indices, importance_weights = memory.sample(batch_size, beta)
    states, actions, rewards, next_states, dones, masks = zip(*batch)
    states = torch.cat(states).to(device, non_blocking=True)
    next_states = torch.cat(next_states).to(device, non_blocking=True)
    # Keeping targets bounded prevents a single checkmate from destabilising Q-values.
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device).clamp(-1.0, 1.0)
    actions = torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
    dones = torch.tensor(dones, dtype=torch.float32, device=device)
    next_masks = torch.cat(masks).to(device, non_blocking=True)
    importance_weights = torch.tensor(importance_weights, dtype=torch.float32, device=device)

    current_q_values = model(states).gather(1, actions).squeeze(1).clamp(-q_value_clip, q_value_clip)
    with torch.no_grad():
        # Double DQN: online model selects a legal action; target model values it.
        next_actions = model(next_states, mask=next_masks).argmax(dim=1, keepdim=True)
        max_next_q = target_model(next_states).gather(1, next_actions).squeeze(1)
        target_q_values = (rewards + gamma * max_next_q * (1 - dones)).clamp(-q_value_clip, q_value_clip)

    td_errors = target_q_values - current_q_values
    loss = (importance_weights * loss_fn(current_q_values, target_q_values)).mean()
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    memory.update_priorities(indices, td_errors.detach().abs().cpu().tolist())
    return loss.item()


def evaluate(model, engine, args):
    """Play greedy games only; these metrics exclude exploratory random moves."""
    results = {"win": 0, "draw": 0, "loss": 0}
    moves_survived = []
    model.eval()
    try:
        for _ in range(args.evaluation_games):
            board, ai_moves = chess.Board(), 0
            while not board.is_game_over(claim_draw=True):
                state, mask = board_to_tensor(board), get_legal_action_mask(board)
                move, _ = select_action(model, board, state, mask, epsilon=0.0)
                board.push(move)
                ai_moves += 1
                if not board.is_game_over(claim_draw=True):
                    board.push(opponent_move(board, engine, args))
            results[outcome_name(board)] += 1
            moves_survived.append(ai_moves)
    finally:
        model.train()
    return results, sum(moves_survived) / len(moves_survived)


def opponent_move(board, engine, args):
    """Select a reply from either a learning-friendly random opponent or Stockfish."""
    if args.opponent == "random":
        return random.choice(list(board.legal_moves))
    return engine.play(board, chess.engine.Limit(depth=args.stockfish_depth)).move


def run_training_loop(args):
    print(f"Training on {device}.")
    model = ChessDQN().to(device)
    target_model = ChessDQN().to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.SmoothL1Loss(reduction="none")
    memory = ReplayBuffer(capacity=args.memory_size, prioritized=args.prioritized_replay,
                          alpha=args.priority_alpha, max_priority=args.max_priority)
    start_episode, epsilon = 0, args.epsilon
    checkpoint_path = Path(args.checkpoint_dir) / "latest.pt"
    if args.resume:
        start_episode, epsilon = load_checkpoint(Path(args.resume), model, optimizer)
        target_model.load_state_dict(model.state_dict())
        print(f"Resumed checkpoint at episode {start_episode}.")

    writer = SummaryWriter(args.log_dir)
    outcomes = {"win": 0, "draw": 0, "loss": 0}
    environment_steps = 0
    try:
        engine_context = (
            chess.engine.SimpleEngine.popen_uci(args.stockfish_path)
            if args.opponent == "stockfish"
            else nullcontext(None)
        )
        with engine_context as engine:
            if engine is not None:
                engine.configure({"Skill Level": args.stockfish_skill})
            opponent_label = "random legal moves" if args.opponent == "random" else (
                f"Stockfish skill {args.stockfish_skill}/20 at depth {args.stockfish_depth}")
            print(f"Training against {opponent_label}.")
            for episode in range(start_episode + 1, args.episodes + 1):
                board = chess.Board()
                episode_reward, total_loss, updates, ai_moves = 0.0, 0.0, 0, 0

                while not board.is_game_over(claim_draw=True):
                    board_before_turn = board.copy(stack=False)
                    state = board_to_tensor(board)
                    mask = get_legal_action_mask(board)
                    move, action = select_action(model, board, state, mask, epsilon)

                    board.push(move)
                    ai_moves += 1
                    if not board.is_game_over(claim_draw=True):
                        reply = opponent_move(board, engine, args)
                        board.push(reply)

                    done = board.is_game_over(claim_draw=True)
                    reward = calculate_transition_reward(board_before_turn, board)
                    next_state = board_to_tensor(board)
                    next_mask = get_legal_action_mask(board)
                    memory.push(state, action, reward, next_state, done, next_mask)
                    episode_reward += reward
                    environment_steps += 1
                    if environment_steps % args.train_every == 0:
                        progress = min(1.0, environment_steps / args.priority_beta_steps)
                        beta = args.priority_beta_start + (1.0 - args.priority_beta_start) * progress
                        loss = train_step(model, target_model, optimizer, loss_fn, memory,
                                          args.batch_size, args.gamma, beta, args.q_value_clip)
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

                if episode % args.target_update_every == 0:
                    target_model.load_state_dict(model.state_dict())

                if episode % args.evaluate_every == 0:
                    evaluation, average_moves = evaluate(model, engine, args)
                    evaluation_games = sum(evaluation.values())
                    writer.add_scalar("evaluation/win_rate", evaluation["win"] / evaluation_games, episode)
                    writer.add_scalar("evaluation/draw_rate", evaluation["draw"] / evaluation_games, episode)
                    writer.add_scalar("evaluation/loss_rate", evaluation["loss"] / evaluation_games, episode)
                    writer.add_scalar("evaluation/average_moves_survived", average_moves, episode)
                    print(f"  Evaluation (epsilon 0): {evaluation['win']}W/{evaluation['draw']}D/"
                          f"{evaluation['loss']}L | average moves {average_moves:.1f}")

                if episode % args.save_every == 0 or episode == args.episodes:
                    save_checkpoint(checkpoint_path, model, optimizer, episode, epsilon)
                    save_checkpoint(Path(args.checkpoint_dir) / f"episode-{episode}.pt", model, optimizer, episode, epsilon)
    finally:
        writer.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--stockfish-path", default="stockfish")
    parser.add_argument("--stockfish-depth", type=int, default=1,
                        help="Stockfish search depth; start at 1 while the agent is learning.")
    parser.add_argument("--stockfish", "--stockfish-skill", dest="stockfish_skill", type=int, default=1,
                        choices=range(21), help="Stockfish strength from 0 (weakest) to 20 (strongest).")
    parser.add_argument("--opponent", choices=("stockfish", "random"), default="stockfish",
                        help="Use random first to teach basic play, then Stockfish to improve it.")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--log-dir", default="runs/chess-dqn")
    parser.add_argument("--resume", help="Checkpoint path to resume from.")
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--memory-size", type=int, default=50_000)
    parser.add_argument("--prioritized-replay", action=argparse.BooleanOptionalAction, default=True,
                        help="Prioritize transitions with high TD error (enabled by default).")
    parser.add_argument("--priority-alpha", type=float, default=0.6)
    parser.add_argument("--priority-beta-start", type=float, default=0.4)
    parser.add_argument("--priority-beta-steps", type=int, default=100_000)
    parser.add_argument("--max-priority", type=float, default=10.0,
                        help="Cap replay priority so one bad transition cannot dominate training.")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Experiences per GPU update; 256 is a good T4 starting point.")
    parser.add_argument("--train-every", type=int, default=4,
                        help="Run one batched optimizer update after this many game moves.")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--q-value-clip", type=float, default=10.0,
                        help="Bound online and target Q-values to prevent value divergence.")
    parser.add_argument("--epsilon", type=float, default=0.5)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--target-update-every", type=int, default=10,
                        help="Copy learned weights to the stable target network every N episodes.")
    parser.add_argument("--evaluate-every", type=int, default=25,
                        help="Run greedy, no-exploration evaluation every N episodes.")
    parser.add_argument("--evaluation-games", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    run_training_loop(parse_args())
