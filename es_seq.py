import time
from environments.env_utils import make_env
from core.policy_seq import build_policy
from core.es_algo import train

if __name__ == "__main__":
    env_name = "CartPole-v1"
    env = make_env(env_name)
    hidden_sizes = [64, 64]
    policy = build_policy(env, hidden_sizes)
    sigma = 0.1
    alpha = 0.03
    iterations = 500
    num_samples = 384
    max_steps = None
    weight_decay = 0
    
    start = time.time()
    train(env_name, policy, sigma, alpha, iterations, num_samples, max_steps, weight_decay)
    end = time.time()
    elapsed = end - start

    with open("time_log.txt", "a") as f:
        f.write(f"Training time: {elapsed:.2f} seconds\n")

    print(f"Training time: {elapsed:.2f} seconds")
