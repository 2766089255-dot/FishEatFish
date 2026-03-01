# ai_agent.py
# DQN Agent with experience replay, target network, and Double DQN

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from dqn_model import DQN
from replay_buffer import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.998,
                 buffer_capacity=10000, batch_size=64, target_update=100):
        """
        DQN Agent.
        Args:
            state_dim: dimension of state vector
            action_dim: number of discrete actions
            lr: learning rate
            gamma: discount factor
            epsilon: initial exploration rate
            epsilon_min: minimum exploration rate
            epsilon_decay: decay factor per step
            buffer_capacity: size of replay buffer
            batch_size: mini-batch size for training
            target_update: frequency of target network update (in steps)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.update_counter = 0

        self.policy_net = DQN(state_dim, action_dim).to(device)
        self.target_net = DQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_capacity)

    def select_action(self, state, eval_mode=False):
        """
        Choose action using epsilon-greedy policy.
        If eval_mode=True, always choose greedy action (no exploration).
        """
        if not eval_mode and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        """Save a transition in the replay buffer."""
        self.memory.push(state, action, reward, next_state, done)

    def update(self):
        """Perform one training step using a mini-batch."""
        if len(self.memory) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        # Double DQN: use policy net to select best next action, target net to evaluate
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1).unsqueeze(1)
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * next_q

        current_q = self.policy_net(states).gather(1, actions)
        loss = F.mse_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network periodically
        self.update_counter += 1
        if self.update_counter % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path):
        """Save policy network weights."""
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        """Load policy network weights and sync target network."""
        self.policy_net.load_state_dict(torch.load(path, map_location=device))
        self.target_net.load_state_dict(self.policy_net.state_dict())