from core.policy_seq import get_flat_params, set_flat_params
import torch

def sample_antithetic_perturbation(num_samples, sigma, env, policy, max_steps, rollout_fn):
    rewards = []
    noises = []
    theta = get_flat_params(policy)
    for _ in range(num_samples):
        eps = torch.randn_like(theta)
        for sign in [1, -1]:
            perturbed = theta + sign * (sigma * eps)
            set_flat_params(policy, perturbed)
            reward, _ = rollout_fn(policy, env, max_steps)
            rewards.append(reward)
            set_flat_params(policy, theta)
        noises.append(eps)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    noises = torch.stack(noises)
    return rewards, noises
