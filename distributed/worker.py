import ray
from core.policy import Policy, get_flat_params, set_flat_params
from environments.env_utils import make_env, extract_envs_info
from core.rollout import rollout

@ray.remote
class ESWorker:
    def __init__(self, env_names, noise, hidden_sizes, shared_output):
        self.envs = [make_env(env_name) for env_name in env_names]
        input_dims, output_dims, output_activation = extract_envs_info(self.envs)
        self.policy = Policy(input_dims, hidden_sizes, output_dims, output_activation, shared_output)
        self.noise = noise
        self.param_dims = [get_flat_params(self.policy, i).shape[0] for i in range(len(env_names))]

    def set_policy(self, theta, task_id):
        set_flat_params(self.policy, theta, task_id)

    def evaluate(self, task_id, theta, sigma, batch_size, max_steps=None):
        all_rewards = []
        all_steps = []
        indices = []

        for _ in range(batch_size):
            idx = self.noise.sample_index()
            eps = self.noise.get(idx, self.param_dims[task_id])
            indices.append(idx)

            self.set_policy(theta + sigma * eps, task_id)
            reward, steps = rollout(self.policy, self.envs, task_id, max_steps)
            all_rewards.append(reward)
            all_steps.append(steps)

            self.set_policy(theta, task_id)

            self.set_policy(theta - sigma * eps, task_id)
            reward, steps = rollout(self.policy, self.envs, task_id, max_steps)
            all_rewards.append(reward)
            all_steps.append(steps)

            self.set_policy(theta, task_id)

        return indices, all_rewards, all_steps
