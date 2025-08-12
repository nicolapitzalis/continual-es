from functional.utils import compute_centered_ranks

def compute_gradient(rewards, noises, sigma, rank=True):
    dim = noises.shape[0]
    rewards = rewards.view(dim, 2)
    paired = rewards[:, 0] - rewards[:, 1]
    if rank:
        paired = compute_centered_ranks(paired)
    return (paired @ noises) / (dim * sigma)

def apply_weight_decay(theta, weight_decay):
    return theta * (1.0 - weight_decay)
