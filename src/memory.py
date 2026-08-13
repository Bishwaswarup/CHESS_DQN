from collections import deque
import random


class ReplayBuffer:
    def __init__(self, capacity=50000, prioritized=True, alpha=0.6):
        # A deque automatically pushes old memories out when it reaches capacity
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.prioritized = prioritized
        self.alpha = alpha

    def push(self, state, action_idx, reward, next_state, done, mask):
        """Save experience in CPU RAM; keep VRAM available for batched training."""
        self.buffer.append((
            state.detach().cpu(), action_idx, reward,
            next_state.detach().cpu(), done, mask.detach().cpu(),
        ))
        self.priorities.append(max(self.priorities, default=1.0))

    def sample(self, batch_size, beta=0.4):
        """Sample high-TD-error experiences more often, with bias correction."""
        if not self.prioritized:
            indices = random.sample(range(len(self.buffer)), batch_size)
            return [self.buffer[index] for index in indices], indices, [1.0] * batch_size
        scaled = [priority ** self.alpha for priority in self.priorities]
        total = sum(scaled)
        probabilities = [priority / total for priority in scaled]
        indices = random.choices(range(len(self.buffer)), weights=probabilities, k=batch_size)
        weights = [(len(self.buffer) * probabilities[index]) ** (-beta) for index in indices]
        maximum = max(weights)
        return [self.buffer[index] for index in indices], indices, [weight / maximum for weight in weights]

    def update_priorities(self, indices, priorities):
        for index, priority in zip(indices, priorities):
            self.priorities[index] = max(float(priority), 1e-5)

    def __len__(self):
        return len(self.buffer)
