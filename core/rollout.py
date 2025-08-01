import torch
import gymnasium as gym
from core.policy import set_flat_params

def rollout(policy, env, max_steps=None):
    obs, _ = env.reset(seed=None)
    total_reward = 0.0
    steps = 0
    while True:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = policy(obs_tensor).squeeze()
            if isinstance(env.action_space, gym.spaces.Discrete):
                action = torch.argmax(output).item()
            else:
                action = output.numpy()
                action = action.clip(env.action_space.low, env.action_space.high)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        if terminated or truncated or (max_steps is not None and steps >= max_steps):
            break
    return total_reward, steps

