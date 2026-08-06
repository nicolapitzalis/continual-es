import gymnasium as gym
import numpy as np
import torch


def rollout(
    policy,
    envs,
    task_id,
    action_bins,
    max_steps=None,
    seed=None,
    obs_mean=None,
    obs_std=None,
    obs_clip=5.0,
    collect_observations=False,
):
    env = envs[task_id]
    observation, _ = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    obs_sum = np.zeros(env.observation_space.shape, dtype=np.float64)
    obs_sumsq = np.zeros(env.observation_space.shape, dtype=np.float64)
    obs_count = 0

    while True:
        if collect_observations:
            obs_sum += observation
            obs_sumsq += np.square(observation)
            obs_count += 1

        policy_observation = observation
        if obs_mean is not None and obs_std is not None:
            policy_observation = np.clip(
                (observation - obs_mean) / obs_std,
                -obs_clip,
                obs_clip,
            )

        obs_tensor = torch.as_tensor(
            policy_observation,
            dtype=torch.float32,
        ).unsqueeze(0)
        with torch.no_grad():
            output = policy(obs_tensor, task_id).squeeze(0)
            action = output_to_action(output, env.action_space, action_bins)

        observation, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        if (
            terminated
            or truncated
            or (max_steps is not None and steps >= max_steps)
        ):
            break

    return total_reward, steps, obs_sum, obs_sumsq, obs_count


def output_to_action(output, action_space, action_bins):
    if isinstance(action_space, gym.spaces.Discrete):
        return torch.argmax(output).item()

    if action_bins == 1:
        action = output.cpu().numpy()
        return action.clip(action_space.low, action_space.high)

    action_dim = action_space.shape[0]
    logits = output.reshape(action_dim, action_bins)
    bin_indices = logits.argmax(dim=-1)
    low = torch.as_tensor(
        action_space.low,
        dtype=output.dtype,
        device=output.device,
    )
    high = torch.as_tensor(
        action_space.high,
        dtype=output.dtype,
        device=output.device,
    )
    fraction = bin_indices.to(output.dtype) / (action_bins - 1)
    return (low + fraction * (high - low)).cpu().numpy()


def evaluate_policy(
    policy,
    envs,
    task_id,
    action_bins,
    episode_seeds,
    obs_mean=None,
    obs_std=None,
    obs_clip=5.0,
):
    rewards = []
    for episode_seed in episode_seeds:
        reward, _, _, _, _ = rollout(
            policy,
            envs,
            task_id,
            action_bins,
            max_steps=None,
            seed=episode_seed,
            obs_mean=obs_mean,
            obs_std=obs_std,
            obs_clip=obs_clip,
            collect_observations=False,
        )
        rewards.append(reward)
    return sum(rewards) / len(rewards) if rewards else 0.0

