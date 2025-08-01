import ray
import torch
from core.policy import get_flat_params, set_flat_params, build_policy, extract_and_transfer_hidden
from environments.env_utils import make_env
from core.rollout import rollout

@ray.remote
class ESWorker:
    def __init__(self, env_names, noise, hidden_sizes=[64, 64]):
        self.envs = [make_env(env_name) for env_name in env_names]
        self.env = self.envs[0]
        self.policies = [build_policy(env, hidden_sizes=hidden_sizes) for env in self.envs]
        self.policy = self.policies[0]
        self.noise = noise
        self.param_dim = get_flat_params(self.policy).shape[0]

    def set_policy(self, theta):
        set_flat_params(self.policy, theta)

    def evaluate(self, theta, sigma, batch_size, max_steps=None):
        all_rewards = []
        all_steps = []
        indices = []

        for _ in range(batch_size):
            idx = self.noise.sample_index()
            eps = self.noise.get(idx, self.param_dim)
            indices.append(idx)

            self.set_policy(theta + sigma * eps)
            reward, steps = rollout(self.policy, self.env, max_steps)
            all_rewards.append(reward)
            all_steps.append(steps)

            self.set_policy(theta)

            self.set_policy(theta - sigma * eps)
            reward, steps = rollout(self.policy, self.env, max_steps)
            all_rewards.append(reward)
            all_steps.append(steps)

            self.set_policy(theta)

        return indices, all_rewards, all_steps
    
    def set_env(self, theta, current_env, next_env):
        for i, env in enumerate(self.envs):
            if env.spec.id == next_env:
                self.env = env
                extract_and_transfer_hidden(theta, current_env, self.policies[i])
                self.policy = self.policies[i]
                self.param_dim = get_flat_params(self.policy).shape[0]
                return get_flat_params(self.policy)
        raise ValueError(f"Environment {env_name} not found in worker's environments.")