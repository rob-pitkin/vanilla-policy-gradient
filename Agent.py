import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Any, List
from collections import deque


class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
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

    def select_action(self, state) -> (int, Any):
        """
        Selects an action given the current state

        Args:
            state (np.ndarray): the current state

        Returns:
            int: the action to take
            torch.Tensor: the log probability of the action
        """
        input_tensor = torch.tensor(state, dtype=torch.float32)
        output_probs = self.forward(input_tensor)
        dist = torch.distributions.Categorical(output_probs)
        action = dist.sample()
        return action.item(), distribution.log_prob(action)


class BaselineNetwork(nn.Module):
    def __init__(self, discount_factor=1.0):
        super(BaselineNetwork, self).__init__()
        self.state_dim = 4
        self.value_dim = 1
        self.hidden_dim = 64
        self.ff1 = nn.Linear(self.state_dim, self.hidden_dim)
        self.ff2 = nn.Linear(self.hidden_dim, self.value_dim)
        self.activation = nn.ReLU()
        self.discount_factor = discount_factor

    def forward(self, x):
        x = self.ff1(x)
        x = self.activation(x)
        x = self.ff2(x)
        return x


class PolicyGradientAgent:
    def __init__(self):
        self.policy = PolicyNetwork()
        self.baseline = BaselineNetwork()
        self.learning_rate = 1e-4

    def calculate_return(self, rewards: List, discount_factor: float) -> List:
        """
        Calculates the discounted returns from a list of rewards

        Args:
            rewards (List): The list of rewards from an episode
            discount_factor (float): The discount factor

        Returns:
            torch.Tensor: a list of discounted returns
        """
        returns = deque()
        running_return = 0
        for i, r in enumerate(reversed(rewards)):
            running_return += r * (discount_factor**i)
            returns.appendleft(running_return)
        return list(returns)

    def calculate_policy_loss(self, log_probs, returns, baseline_values):
        loss = 0
        for log_prob, r, b in zip(
            torch.tensor(log_probs),
            torch.tensor(returns),
            torch.tensor(baseline_values),
        ):
            loss -= log_prob * (r - b)
        return loss

    def calculate_baseline_loss(self, returns, baseline_values):
        loss = nn.MSELoss()
        return loss(torch.tensor(baseline_values), torch.tensor(returns))

    def train(self, env, num_episodes):
        policy_optim = optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        baseline_optim = optim.Adam(self.baseline.parameters(), lr=self.learning_rate)

        for i in range(num_episodes):
            # Initialize lists for tracking rewards, action probabilites, and baseline values
            rewards = []
            log_probs = []
            baseline_values = []

            # Reset the environment
            obs, info = env.reset()
            episode_over = False

            while not episode_over:
                # Forward passes of the policy and baseline network
                state = torch.tensor(obs)
                action, log_prob = self.policy.select_action(state)
                baseline_value = self.baseline.forward(state)

                # Taking action a in the environment
                obs, reward, terminated, truncated, info = env.step(action)

                # recording the results
                rewards.append(reward)
                log_probs.append(log_prob)
                baseline_values.append(baseline_value)

                episode_over = terminated or truncated

            # Calculate the returns
            returns = self.calculate_return(rewards, self.discount_factor)

            # Calculate policy loss and backprop
            policy_optim.zero_grad()
            policy_loss = self.calculate_policy_loss(
                log_probs, returns, baseline_values
            )
            policy_loss.backward()
            policy_optim.step()

            # Calculate baseline loss and backprop
            baseline_optim.zero_grad()
            baseline_loss = self.calculate_baseline_loss(returns, baseline_values)
            baseline_loss.backward()
            baseline_optim.step()

            # Print progress every 100 episodes
            if i % 100 == 0:
                print(
                    f"Episode {i}/{num_episodes}, Policy Loss: {policy_loss.item()}, Baseline Loss: {baseline_loss.item()}"
                )

    def evaluate_agent(self, env, num_episodes):
        with torch.no_grad():
            total_rewards = 0
            for _ in range(num_episodes):
                obs, info = env.reset()
                episode_over = False
                while not episode_over:
                    state = torch.tensor(obs)
                    action, _ = self.policy.select_action(state)
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_rewards += reward
            print(
                f"Average reward after {num_episodes} episodes: {total_rewards / num_episodes}"
            )
