import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class PolicyGradientAgent(nn.Module):
    def __init__(self):
        super(PolicyGradientAgent, self).__init__()
        self.state_dim = 4
        self.hidden_dim = 64
        self.action_dim = 2
        self.ff1 = nn.Linear(self.state_dim, self.hidden_dim)
        self.ff2 = nn.Linear(self.hidden_dim, self.action_dim)
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.ff1(x)
        x = self.activation(x)
        x = self.ff2(x)
        return self.softmax(x)


def train_agent(env, agent, num_episodes):
    pass
