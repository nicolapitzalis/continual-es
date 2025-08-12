import torch

# fitness shaping from Wierstra et al. 2014
def fitness_shaping(x):
    n = x.shape[0]
    ranks = torch.argsort(torch.argsort(-x)) + 1  # Rank 1 = best
    ranks = ranks.float()
    log_base = torch.log(torch.tensor(n / 2 + 1.0, device=ranks.device, dtype=ranks.dtype))
    utilities = torch.clamp(log_base - torch.log(ranks), min=0.0)
    utilities /= utilities.sum()
    utilities -= 1.0 / n
    return utilities

# z-score normalization on ranks
def z_score_ranks(x):
    ranks = torch.argsort(torch.argsort(x))
    ranks = ranks.float()
    return (ranks - ranks.mean()) / (ranks.std() + 1e-8)

# centered ranks (values in [-0.5, 0.5])
def compute_centered_ranks(x):
    ranks = torch.argsort(torch.argsort(x))
    ranks = ranks.float()
    return ranks / (len(x) - 1) - 0.5

# normalized reverse ranks (values in [0, 1])
def compute_weighted_ranks(x):
    x = -x
    ranks = torch.argsort(torch.argsort(x))
    ranks = ranks.float()
    return (len(x) - 1 - ranks).float() / (len(x) - 1)
