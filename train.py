import torch
from Agent import PolicyGradientAgent
import gymnasium as gym


def main():
    env = gym.make("CartPole-v1")
    num_episodes = 100000
    agent = PolicyGradientAgent()
    agent.train(env, num_episodes, True)
    show_env = gym.make("CartPole-v1", render_mode="human")
    agent.evaluate_agent(show_env, 10)


if __name__ == "__main__":
    main()
