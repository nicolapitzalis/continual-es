import torch
import torch.nn.functional as F
import gymnasium as gym
from core.rollout import rollout

def make_env(env_name, seed=None, render_mode=None):
    env = gym.make(env_name, render_mode=render_mode)
    if seed is not None:
        env.reset(seed=seed)
    return env

def extract_envs_info(envs):
    input_dims = [env.observation_space.shape[0] for env in envs]
    output_dims = [env.action_space.n if hasattr(env.action_space, 'n') else env.action_space.shape[0] for env in envs]
    activation = [torch.tanh if hasattr(env.action_space, 'shape') else lambda x: F.softmax(x, dim=-1) for env in envs]
    return input_dims, output_dims, activation

def eval_policy(policy, envs, task_id, max_steps=None, iterations=10):
    rewards = []
    for _ in range(iterations):
        rew, _ = rollout(policy, envs, task_id, max_steps=max_steps)
        rewards.append(rew)
    return sum(rewards) / len(rewards) if rewards else 0
