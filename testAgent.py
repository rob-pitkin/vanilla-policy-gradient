import gymnasium as gym


def main():
    env = gym.make("CartPole-v1", render_mode="human")
    obs, info = env.reset()

    episode_over = False
    while not episode_over:
        action = (
            env.action_space.sample()
        )  # sample a random action from the action space
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Observation: {obs}, reward: {reward}")

        episode_over = terminated or truncated


if __name__ == "__main__":
    main()
