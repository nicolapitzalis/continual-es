from core.policy import build_policy
from core.es_algo import train
from environments.env_utils import make_env
import time

if __name__ == "__main__":
    env_name = "CartPole-v1"
    env = make_env(env_name)
    hidden_sizes = [64, 64]
    policy = build_policy(env, hidden_sizes)
    sigma = 0.1
    alpha = 0.03
    iterations = 300
    num_samples = 384
    max_steps = None
    weight_decay = 0.005
    
    start = time.time()
    train(env_name, policy, sigma, alpha, iterations, num_samples, max_steps, weight_decay)
    end = time.time()
    print(f"Training time: {end - start:.2f} seconds")