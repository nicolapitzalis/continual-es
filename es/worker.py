import numpy as np
import ray

from es.envs import extract_envs_info, make_env
from es.policy import Policy, get_flat_params, set_flat_params
from es.rollout import rollout


@ray.remote(num_cpus=1)
class ESWorker:
    def __init__(
        self,
        env_names,
        noise,
        hidden_sizes,
        shared_output,
        action_bins,
        normalize_observations,
        obs_clip,
    ):
        self.envs = [make_env(env_name) for env_name in env_names]
        self.input_dims, output_dims, output_activations = extract_envs_info(
            self.envs,
            action_bins,
        )
        self.policy = Policy(
            self.input_dims,
            hidden_sizes,
            output_dims,
            output_activations,
            shared_output,
        )
        self.noise = noise
        self.action_bins = action_bins
        self.normalize_observations = normalize_observations
        self.obs_clip = obs_clip
        self.param_dims = [
            get_flat_params(self.policy, task_id).shape[0]
            for task_id in range(len(env_names))
        ]

    def set_policy(self, theta, task_id):
        set_flat_params(self.policy, theta, task_id)

    def evaluate(
        self,
        task_id,
        theta,
        sigma,
        indices,
        max_steps,
        episode_seeds,
        obs_mean,
        obs_std,
    ):
        all_rewards = []
        all_steps = []
        obs_sum = np.zeros(self.input_dims[task_id], dtype=np.float64)
        obs_sumsq = np.zeros(self.input_dims[task_id], dtype=np.float64)
        obs_count = 0

        for noise_index, episode_seed in zip(indices, episode_seeds):
            epsilon = self.noise.get(noise_index, self.param_dims[task_id])

            for direction in (1, -1):
                self.set_policy(theta + direction * sigma * epsilon, task_id)
                reward, steps, roll_sum, roll_sumsq, roll_count = rollout(
                    self.policy,
                    self.envs,
                    task_id,
                    self.action_bins,
                    max_steps,
                    episode_seed,
                    obs_mean=obs_mean,
                    obs_std=obs_std,
                    obs_clip=self.obs_clip,
                    collect_observations=self.normalize_observations,
                )
                all_rewards.append(reward)
                all_steps.append(steps)
                obs_sum += roll_sum
                obs_sumsq += roll_sumsq
                obs_count += roll_count

        self.set_policy(theta, task_id)
        return indices, all_rewards, all_steps, obs_sum, obs_sumsq, obs_count

