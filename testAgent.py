import gymnasium as gym
import torch
import Agent
from Agent import PolicyGradientAgent
import unittest


def testCartPole():
    env = gym.make("CartPole-v1", render_mode="human")
    obs, info = env.reset()
    print(
        f"Observation space shape: {obs.shape}, action space shape: {env.action_space}"
    )

    episode_over = False
    while not episode_over:
        action = (
            env.action_space.sample()
        )  # sample a random action from the action space
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Observation: {obs}, action: {action}, reward: {reward}")

        episode_over = terminated or truncated


class TestPolicyGradientAgent(unittest.TestCase):
    def testPolicyNetworkForward(self):
        agent = PolicyGradientAgent()
        env = gym.make("CartPole-v1")
        obs, info = env.reset()
        out = agent.policy.forward(torch.tensor(obs))
        print(f"Example forward from policy network: {out}")
        self.assertTrue(sum(out) == 1.0)

    def testCalculateReturn(self):
        rewards = [i for i in range(10)]
        expected_return = [45, 45, 44, 42, 39, 35, 30, 24, 17, 9]
        agent = PolicyGradientAgent()
        returns = agent.calculate_return(rewards, 1.0)
        print(returns)
        for i, r in enumerate(returns):
            self.assertTrue(expected_return[i] == r)


def main():
    # testCartPole()
    pass


if __name__ == "__main__":
    main()
    unittest.main()
