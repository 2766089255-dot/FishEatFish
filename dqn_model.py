# dqn_model.py
# Deep Q-Network definition using PyTorch

import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    """Fully connected neural network for Q-value approximation."""
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)