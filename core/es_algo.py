from core.policy_seq import get_flat_params, set_flat_params
from core.rollout_seq import rollout
from environments.env_utils import make_env
from core.perturbation import sample_antithetic_perturbation
from core.updates import compute_gradient_seq, apply_weight_decay
from functional.utils import compute_centered_ranks
import time

def train(env_name, policy, sigma, alpha, iterations, num_samples, max_steps, weight_decay=0.005):
    env = make_env(env_name)
    theta = get_flat_params(policy)
    for t in range(iterations):
        start_time = time.time()
        rewards, noises = sample_antithetic_perturbation(num_samples, sigma, env, policy, max_steps, rollout)
        print(f"Iter {t+1:03d}: avg raw reward = {rewards.mean().item():.2f}")

        grad_estimate = compute_gradient_seq(rewards, noises, sigma, rank=True)

        theta += alpha * grad_estimate
        theta = apply_weight_decay(theta, weight_decay)
        set_flat_params(policy, theta)

        end_time = time.time()
        print(f"Iter {t+1:03d}: time = {end_time - start_time:.2f}s")
