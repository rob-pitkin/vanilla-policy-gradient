import gymnasium as gym
import torch
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
    def testAgentForward(self):
        agent = PolicyGradientAgent()
        env = gym.make("CartPole-v1")
        obs, info = env.reset()
        out = agent.forward(torch.tensor(obs))
        print(out)
        self.assertTrue(sum(out) == 1.0)


def main():
    # testCartPole()
    pass


if __name__ == "__main__":
    main()
    unittest.main()
