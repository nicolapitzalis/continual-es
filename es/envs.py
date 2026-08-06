import gymnasium as gym
import torch
import torch.nn.functional as F


def identity(x):
    return x


def softmax_last(x):
    return F.softmax(x, dim=-1)


def make_env(env_name, seed=None, render_mode=None):
    env = gym.make(env_name, render_mode=render_mode)
    if seed is not None:
        env.reset(seed=seed)
    return env


def extract_envs_info(envs, action_bins=1):
    if action_bins < 1:
        raise ValueError("action_bins must be at least 1")

    input_dims = [env.observation_space.shape[0] for env in envs]
    output_dims = []
    output_activations = []

    for env in envs:
        action_space = env.action_space
        if isinstance(action_space, gym.spaces.Discrete):
            output_dims.append(action_space.n)
            output_activations.append(softmax_last)
        elif isinstance(action_space, gym.spaces.Box):
            action_dim = action_space.shape[0]
            output_dims.append(
                action_dim if action_bins == 1 else action_dim * action_bins
            )
            output_activations.append(
                torch.tanh if action_bins == 1 else identity
            )
        else:
            raise TypeError(
                f"Unsupported action space: {type(action_space).__name__}"
            )

    return input_dims, output_dims, output_activations

