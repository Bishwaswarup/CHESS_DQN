import torch
import torch.nn as nn
import torch.nn.functional as F


class ChessDQN(nn.Module):
    def __init__(self):
        super().__init__()

        # Input shape: (Batch, 18, 8, 8)
        self.conv1 = nn.Conv2d(18, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        # Dueling heads separate state value from move advantage, which is
        # useful when many legal moves lead to similar positions.
        self.value = nn.Linear(512, 1)
        self.advantage = nn.Linear(512, 4096)

    def forward(self, x, mask=None):
        # Pass data through Conv -> BatchNorm -> ReLU
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        q_values = self.value(x) + self.advantage(x) - self.advantage(x).mean(dim=1, keepdim=True)

        # Mask out illegal moves
        if mask is not None:
            q_values = torch.where(mask == 1.0, q_values, torch.full_like(q_values, -1e9))

        return q_values
