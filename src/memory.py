from collections import deque
import random


class ReplayBuffer:
    def __init__(self, capacity=50000):
        # A deque automatically pushes old memories out when it reaches capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action_idx, reward, next_state, done, mask):
        """Save experience in CPU RAM; keep VRAM available for batched training."""
        self.buffer.append((
            state.detach().cpu(), action_idx, reward,
            next_state.detach().cpu(), done, mask.detach().cpu(),
        ))

    def sample(self, batch_size):
        """Randomly samples a batch of experiences for training."""
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
